# CS2 · precio antes de salir — qué pidió cada instructivo y cómo lo resolvimos

Guía para la defensa, ordenada **entrega por entrega, siguiendo los instructivos de la cátedra**. En cada entrega va:

1. **Qué pide el instructivo**: la pregunta de fondo, qué hay que traer y cómo se reparten los minutos.
2. **Cómo lo cumplimos**: qué hicimos, con los números y **por qué se decidió así**.
3. **Lo que hay que saber explicar**: cada pregunta del instructivo, con su respuesta preparada.
4. **Cómo se aprueba**: la lista del instructivo, punto por punto, con dónde se ve cada cosa.

Los instructivos repiten que **la evaluación es individual** y que **las preguntas van dirigidas**. Este documento es para que cualquiera de los siete pueda contestar sobre cualquier parte.

**De qué fecha son los números:**

| Entrega | Datos |
|---|---|
| Entrega 1 (pipeline) | Corrida del **30/10** que usa la aplicación: 10.003 filas |
| Entrega 2 (EDA) | Snapshot del **15/10**: 2.012 filas |
| Entrega 3 (modelado) y Entrega 4 (aplicación) | Snapshot del **30/10**: 10.003 filas |

---

## 0 · El hilo de todo el proyecto

**Pregunta:** ¿se puede estimar, **sólo con los atributos que se conocen el día del lanzamiento**, en qué **percentil de precio** va a quedar un skin de arma nuevo de CS2?

Casi todas las decisiones de las cuatro entregas salen de tomarse en serio esa frase:

- **"Lo que se conoce el día del lanzamiento"**: lleva al **chequeo de fuga**. Por eso `sell_listings` y la antigüedad no entran.
- **"Un skin nuevo"**: la validación tiene que imitar el uso. Se entrena con colecciones viejas y se prueba con colecciones que el modelo no vio, en el tiempo.
- **"Percentil y no precio"**: el nivel del mercado se mueve, pero el orden casi no.

Y una idea de método que aparece en todas partes: **comparar cada cosa consigo misma** (en pares). Primero arregló las hipótesis de la Entrega 2. Después se usó para elegir modelo, columnas e hiperparámetros en la Entrega 3.

---

# ENTREGA 1 · El pipeline

## 1.1 · Qué pide el instructivo

> *«¿Tenés un pipeline que produce tu dataset sin que nadie lo toque a mano?»*

**Qué hay que traer:**

- Airflow levantado, con la interfaz abierta.
- Una corrida en verde ya hecha, visible en el historial. No se corre en vivo.
- El CSV que produjo esa corrida.
- Los siete criterios verificados, con los números a mano.

**Los 20 minutos:**

| Min. | Momento | Qué decimos nosotros |
|---|---|---|
| 0–2 | De qué se trata | La pregunta en una frase. La fuente: el Steam Community Market (precios) y el catálogo de ByMykel (atributos). Y «una fila es un skin de arma de CS2 (arma + diseño + exterior + variante StatTrak/Souvenir), observado en un snapshot de precio de Steam» |
| 2–7 | La corrida | Recorrer el grafo de las 7 tareas (1.2) |
| 7–12 | El dataset | Los siete criterios (1.3) |
| 12–18 | Preguntas dirigidas | 1.4 y 1.5 |

## 1.2 · El grafo: qué hace cada tarea y por qué está donde está

El archivo es `pipeline/cs2_pipeline.py`, un DAG de Airflow (Astro) con arquitectura **medallion**:

- **Bronce:** el dato crudo, comprimido, tal como llegó, en `bronze/snapshot=<fecha>/`.
- **Plata:** el dato limpio y tipado.
- **El CSV publicado.**

| # | Tarea | Qué hace | Por qué está ahí |
|---|---|---|---|
| T1 | `descargar_catalogo` | Baja del repositorio de ByMykel `skins_not_grouped.json`, `collections.json` y `crates.json` al bronce, sin interpretarlos | Es la fuente 2. El catálogo de skins **no trae la colección** (se verificó sobre los 21.407 ítems): la colección y la caja viven en los otros dos archivos |
| T2 | `descargar_precios_steam` | Pide al mercado de Steam una categoría de arma por vez (`category_730_Type[]`) y guarda cada página en el bronce | Es la fuente 1. Va **en paralelo** con T1 porque no dependen una de la otra. Pide categoría por categoría porque, sin el filtro, Steam devuelve el mercado entero (35.449 ítems) y sólo ~35 % sobrevive a la unión |
| T3 | `procesar_atributos` | Pasa el catálogo de bronce a plata: sólo armas, rareza y exterior ordinales, **colección cruzada por `skin_id`** (no por nombre), caja, `origen` y `p_item` | Depende sólo del bronce del catálogo. Cruzar por `skin_id` da 99,93 % de cobertura y una sola colección por skin |
| T4 | `procesar_precios` | Pasa las páginas de Steam de bronce a plata, saca duplicados y precios ≤ 0, y cuenta cuántos ítems crudos llegaron | Depende sólo del bronce de precios. Ese conteo es el que usa T5 para detectar una descarga cortada |
| T5 | `unir_y_validar` | Une las dos platas por `market_hash_name` y valida. **Si algo falla, no se publica nada** | Necesita las dos ramas. Exige el 95 % de lo planificado, medido sobre los ítems crudos, y que el 95 % de las filas tenga colección |
| T6 | `usar_snapshot_previo` | **Rama de respaldo** (`trigger_rule="one_failed"`): si algo de la rama principal falló, publica el último snapshot bueno | Depende de **todas** las tareas anteriores. Si no, `one_failed` nunca se cumpliría y el respaldo quedaría decorativo. Si no hay respaldo se saltea, así el error real queda a la vista |
| T7 | `generar_calidad` | (`one_success`) Mide los siete criterios sobre el CSV publicado y los escribe en `calidad_<fecha>.json` | El instructivo pide llegar con los números a mano. Quedan en el log de la corrida y en un JSON |

**Por qué T5 mide la completitud en relación con lo planificado y no con un piso fijo.** La falla más peligrosa del scraping es la silenciosa: una página vacía antes de tiempo o un error 429 que corta la descarga. En ese caso el DAG terminaría en verde y publicaría un CSV truncado, **que es peor que no publicar nada**. Se mide sobre los ítems crudos, no sobre las filas finales, porque los filtros sacan filas a propósito y un filtro que funciona bien no tiene que parecer una descarga rota.

**Configuración del DAG:** `schedule="@daily"`, `retries=2` y `max_active_runs=1`. Este último evita que dos corridas escriban el mismo bronce a la vez, y fue una de las correcciones posteriores.

**Por qué es una muestra y cómo se armó.**

- Steam pagina **de a 10 ítems** (ignora el parámetro `count`) y limita por IP. Bajar los ~12.400 ítems de arma serían ~1.240 pedidos y más de 5 horas.
- Por eso el parámetro `filas_objetivo` reparte las páginas entre las categorías **en proporción a su tamaño**, y dentro de cada una **a lo largo de todo el catálogo**: es un muestreo sistemático.
- Steam ordena por nombre, así que pedir "las primeras N páginas" sería pedir el principio del alfabeto. Con la muestra sistemática eso no pasa.
- La corrida del 30/10 trajo **10.003 filas, el 80,7 % de los 12.399 ítems de arma del mercado**, y el JSON de calidad lo declara como muestra.

**El embudo, paso a paso** (corrida del 30/10, el mismo que muestra la aplicación):

| Paso | Sale | Quedan |
|---|---|---|
| Ítems de arma publicados en Steam ese día | — | 12.399 |
| Muestra sistemática (1.003 páginas de 10) | — | 10.030 |
| Duplicados por paginar con el mercado en vivo (un ítem puede aparecer en dos páginas) | −27 | 10.003 |
| Sin precio o con precio ≤ 0 | −0 | 10.003 |
| Unión con el catálogo de ByMykel (16.837 ítems de arma) | −0 | 10.003 |
| Validación: clave única, colección en el 100 %, más de 1.000 filas | 0 | 10.003 |

## 1.3 · Los siete criterios, con los números

Son los de `data/calidad_2026-10-30.json`:

| Criterio | Qué pide | Qué da | Cómo se lee |
|---|---|---|---|
| Clave sin duplicados | `True` | **`market_hash_name` es única: True** | Es el test de la unidad de análisis: un nombre de mercado es exactamente un skin en una versión |
| Volumen | más de 1.000 filas | **10.003** | |
| Ancho | 5 o más columnas útiles | **29 columnas** | |
| Mezcla de tipos | numéricas, categóricas y fechas | **12 object, 9 float, 3 datetime, 3 int, 2 bool** | |
| Nulos conocidos | saber cuáles y por qué | `neg_log_p_item` y `p_item` 50,5 %; `caja`, `caja_fecha` y `caja_discontinuada` 47,9 %; `collection_release` y `antiguedad_col_dias` 0,28 % | Ver 1.4 |
| Sin columnas vacías | ninguna al 100 % | **ninguna** | Si una columna opcional del catálogo viniera 100 % vacía, se descarta con un aviso |
| Una fila es… | coincidir con la pregunta | un skin de arma en una versión concreta, en un snapshot | La pregunta es sobre skins, así que una fila es un skin |

## 1.4 · Lo que hay que saber explicar (Entrega 1)

**«Una fila es un ___».**
Un skin de arma de CS2 en una versión concreta (arma + diseño + exterior + StatTrak/Souvenir), observado en un snapshot de precio del Steam Community Market. Coincide con la pregunta, que es sobre el precio de un skin.

**Cuál es la columna objetivo.**
El pipeline publica `sell_price_usd`, el precio crudo. El objetivo del modelo es el **percentil de ese precio dentro del día**, y **a propósito no se calcula en el pipeline**: es una decisión de modelado, y para un skin nuevo se mide contra el mercado que ya existe. Se calcula en el notebook.

**Por qué hay nulos donde hay nulos.**

- **Estructurales** (tienen que estar vacíos):
  - `caja`, `caja_fecha` y `caja_discontinuada`: 52 de las 94 colecciones son de mapa o souvenir y no salen de ninguna caja.
  - `p_item`, `p_rareza` y `neg_log_p_item`: sólo se definen para ítems que salen de una caja. Los souvenir vienen de paquetes de torneo, que tienen otra mecánica, y **no inventamos probabilidades**.
- **Faltantes reales:**
  - `collection_release` (0,28 %): el catálogo no publica la fecha de algunas colecciones.
  - `antiguedad_col_dias`: depende de la anterior.
  - `min_float` y `max_float`: el catálogo no cubre el 100 % de los skins.

**Qué pasa si se corre de nuevo.**
El bronce se particiona por **fecha de captura** y una página ya bajada **nunca se vuelve a pedir**. Por eso, **con el mismo snapshot en bronce sale el mismo CSV**, y la segunda corrida del día es instantánea. Corrido otro día produce **otro** snapshot: no es un bug, es que el mercado se movió, y el bronce guarda la evidencia de los dos días. Con el parámetro `snapshot` se reprocesa un bronce viejo sin pedirle nada a Steam.

**Dónde se guarda el dato crudo.**
En `include/bronze/snapshot=<fecha>/`: las páginas de Steam y los tres JSON de ByMykel, comprimidos y sin interpretar. Si mañana descubrimos que interpretábamos mal una columna, se arregla reprocesando el bronce, sin volver a la fuente. El CSV entregado es la plata; el bronce existe.

## 1.5 · Sobre «tidy» y nuestras concesiones

El instructivo pide que, si el dataset se aparta del ideal, **lo expliquemos en lugar de esconderlo**.

- **No hay celdas con varios valores:** cada skin pertenece a una sola colección (verificado al cruzar por `skin_id`).
- **Concesión 1, atributos de colección repetidos.** `collection_release`, `caja`, `caja_fecha` y `caja_discontinuada` se repiten en todas las filas de una colección. Es una tabla unida (desnormalizada) en lugar de una tabla de skins y otra de colecciones. Se eligió así porque el modelo necesita una fila con todo.
- **Concesión 2, es una muestra.** Es el 80,7 % del mercado, y está declarada en el JSON con su método de muestreo.
- **Concesión 3, la categoría «sin identificar».** Agrupa las variantes Souvenir que no se pudieron asociar a ningún paquete. Es una categoría residual, declarada.
- **Supuestos declarados en el JSON:**
  - `p_wear` supone que el float se sortea uniforme dentro del rango del skin. Quedó parcialmente validado y no entra al modelo.
  - `caja_discontinuada` sale de una lista curada externa (steamdb), y coincide exactamente con un corte por fecha.

## 1.6 · Lo que marcó la cátedra en la Entrega 1, y cómo se resolvió

Se presenta en el minuto 0–2 de la Entrega 2:

| Observación | Qué se cambió |
|---|---|
| No sabíamos de qué colección salía cada skin | `collection` se cruza **por `skin_id`** con `collections.json` y `crates.json`. Cobertura 99,93 %, una colección por skin |
| Faltaba una medida de la escasez | `p_item` y `neg_log_p_item`: la probabilidad de que un drop sea ese ítem exacto, con las odds oficiales (79,92 / 15,98 / 3,2 / 0,64 %) aplicadas por peldaño de la colección |
| Las souvenir no tenían lugar en el análisis | Columna `origen`: caja, paquete souvenir, drop de mapa o sin identificar |
| La fecha del snapshot salía corrida un día | `_snapshot()` usa el momento en que arrancó la corrida, en hora argentina, y no el `logical_date` |
| Si el cruce con el catálogo fallaba, se publicaba igual | Ahora la corrida termina en rojo si menos del 95 % de las filas tiene colección |

## 1.7 · Cómo se aprueba (Entrega 1)

| Requisito del instructivo | Dónde está |
|---|---|
| Corrida completa en verde, visible en la interfaz | La del 30/10 (`run_id manual__2026-10-30T11:40`, `datos_frescos: true`) |
| Saber qué hace cada tarea y por qué está ahí | 1.2 |
| El CSV cumple los criterios medibles | 1.3 y `calidad_2026-10-30.json` |
| Saber defender las decisiones de la lista | 1.4 y 1.5 |
| Que no conteste uno solo | Todos leen este documento |

---

# ENTREGA 2 · Análisis exploratorio (`notebooks/EDA_percentil_del_dia.ipynb`)

## 2.1 · Qué pide el instructivo

> *«¿Sabés qué tiene adentro, y podés defender con un número cada decisión que tomaste sobre él?»*

**Qué hay que traer:**

- El notebook ejecutado, sin errores y con las salidas guardadas.
- El perfil del dataset.
- **Cuatro fichas de hipótesis** con los seis campos, el número, la zona y el gráfico.
- La **tabla de columnas candidatas**.
- Lo que nos marcaron en la Entrega 1, resuelto.

**Los 20 minutos:**

| Min. | Qué |
|---|---|
| 0–2 | Qué cambió del pipeline (tabla 1.6), y si cambió la pregunta: el objetivo pasó de bandas de precio a percentil |
| 2–7 | El perfil (2.2 y 2.3) |
| 7–13 | **El docente elige dos de las cuatro fichas** y se recorren enteras (2.5) |
| 13–18 | Preguntas dirigidas (2.7) |

**Qué cambió desde la Entrega 1, en dos oraciones:** «El pipeline ahora sabe de qué colección y de qué caja sale cada skin, cuán probable es sacarlo y cómo se obtiene, y no publica si el cruce falla. El objetivo dejó de ser una banda de precio: es el percentil del precio dentro del día, con regresión.»

## 2.2 · El perfil, con los números que pide el instructivo

Son los del snapshot del 15/10:

| Qué pide | Qué da | Qué decimos |
|---|---|---|
| `df.shape` | **2.012 filas × 31 columnas** | Una fila es un skin de arma en una versión concreta, el 15/10. Sólo armas: Heavy, Pistols, Rifles y SMGs |
| `dtypes.value_counts()` | 11 float, 10 object, 4 datetime, 3 boolean, 3 int | **«Si alguna numérica quedó como texto, y por qué»:** los booleanos (`stattrak`, `souvenir`, `caja_discontinuada`) vuelven del CSV como `object` cuando tienen nulos, y en una columna `object` el `~` devuelve enteros. Se normalizan al cargar. Las cuatro fechas se fuerzan con `parse_dates` |
| Nulos, las tres peores | `p_item` y `neg_log_p_item` 51,6 %; `caja` 48,9 % | Las tres son **estructurales**: esos ítems no salen de una caja (ver 1.4) |
| Constantes | `fecha_snapshot`, `captured_at` | Son constantes **por construcción**, porque es el corte de un día. Salen del modelo |
| Duplicados de la clave | **0** | |
| Distribución del objetivo | **~10 % en cada decil** | Parejo por construcción: **ninguna clase se come todo** (2.3) |
| Asimetría | `sell_listings` 19,4, `sell_price_usd` 5,9, `min_float` 3,7, `p_item` 3,3, `p_wear` 1,1 | La regla: con \|asimetría\| > 1 se transforma con log. **Ninguna de esas entra cruda**: el precio se reemplaza por su percentil, `p_item` se usa como log (`neg_log_p_item`, asimetría 0,10), `sell_listings` sale por fuga, y `min_float` y `p_wear` no entran. Las que entran son ordinales o ya están en log |

**Filas hermanas.** Hay 563 skins base, con una mediana de 4 filas cada uno (los distintos exteriores, StatTrak y Souvenir). Si en la validación un skin cae de un lado y su hermana del otro, el número sale inflado. **Decisión: la validación agrupa por colección.**

**Qué se hace con los nulos estructurales.** No se borran: borrarlos tiraría la mitad del dataset, y justo la que no es de caja. En la Entrega 3 van con `SimpleImputer(add_indicator=True)` dentro del pipeline: se completa el valor y se agrega una columna que dice "este ítem no sale de una caja". El modelo aprende de esa señal.

## 2.3 · La variable objetivo: por qué el percentil (decisión 1)

- **La columna concreta:** `percentil_precio`, calculada agrupando por `fecha_snapshot`: `rank(pct=True) × 100`. Percentil 90 quiere decir que el skin es más caro que el 90 % del mercado ese día.
- **Por qué no el precio en dólares:**
  1. **Cola larga:** asimetría 5,85, y el más caro vale 471 veces la mediana. El log la corrige (asimetría 0,06), así que esto solo no alcanzaría.
  2. **La razón de fondo, la estabilidad.** Entre el 24/09 y el 15/10 (938 skins en común), el **nivel** del mercado bajó **−9,9 %**, pero el **orden** casi no cambió: **Spearman 0,997**, y cada skin se movió en promedio **2,7 puestos**. Un modelo en dólares queda corrido apenas cambia la fecha; uno de percentil, no.
- **Cómo se vuelve a dólares (decisión 3):** con la distribución de precios del día, "percentil 90" es el precio por debajo del cual está el 90 % del mercado. Con el error real de la mejor línea de base (**±17 puntos**):
  - el percentil 50 (4,28 USD) puede ser cualquier precio entre 1,11 y 16,65 USD;
  - el percentil 90 (131,82 USD), cualquiera entre 27 y 2.017 USD.

  **El error en puntos es parejo, pero en dólares crece en la zona cara.** Se muestra así, no se esconde. Por eso la aplicación muestra un rango y no un número.

## 2.4 · El semáforo y la lección de la versión anterior

**Los cortes de la cátedra, sobre el valor absoluto:**

| Plantilla | Medida | Rojo | Amarillo | Verde |
|---|---|---|---|---|
| Comparación de 2 grupos | separación *d* | < 0,2 | 0,2–0,8 | > 0,8 |
| Comparación de n grupos | η² | < 0,05 | 0,05–0,25 | > 0,25 |
| Asociación | correlación | < 0,2 | 0,2–0,6 | > 0,6 |

- **Verde quiere decir "efecto grande", no "la columna entra".** `sell_listings` correlaciona −0,77 (sería verde) y sale por fuga.
- **El signo no cuenta para el color:** −0,700 es verde.

**Por qué las cuatro hipótesis anteriores dieron amarillo.** Comparaban **skins distintos** entre sí. Lo que más mueve el precio (el diseño, la popularidad) no está en ninguna columna, y ese ruido tapaba todo. **La solución fue comparar cada skin consigo mismo:** el mismo skin en dos exteriores, o en versión souvenir y normal. Con el diseño fijo de los dos lados, el ruido se cancela.

## 2.5 · Las cuatro fichas

**Las condiciones del instructivo, cumplidas** (lo verifica la celda 8.1):

| Condición | Cómo se cumple |
|---|---|
| Al menos una de dominio | H1 y H2 |
| Al menos una sobre una columna del modelo | H3 y H4 |
| Al menos una refutada o inconclusa | H4, inconclusa |
| Cada una con número, zona y gráfico | Sí, las cuatro |
| Movimientos justificados por el gráfico, y nunca más de dos | H1 tiene 2, H2 tiene 0, H3 tiene 1 y H4 tiene 1 |

**Los campos de cada ficha:** afirmación, qué esperaba ver, plantilla, medida (número), zona, movimientos y decisión, más el gráfico. La afirmación y lo esperado se escribieron **antes** de correr la medida.

### H1 · El arma explica el precio más allá de la rareza — dominio — **VERDE**

| Campo | Contenido |
|---|---|
| Afirmación | A igual rareza, el arma cambia la posición del skin: la demanda se concentra en AK-47, AWP y M4 |
| Qué esperaba ver | η² alto, y que **se sostenga dentro de cada rareza, por encima del azar** |
| Plantilla | Comparación de n grupos (η²) |
| Medida | **η² = 0,299** (mediana dentro de las rarezas) |
| Zona | **Verde** (> 0,25) |
| Movimiento 1 | **Medir dentro de cada rareza.** En general daba 0,217 (amarillo), pero ese número mezcla el efecto de la rareza (η² 0,408): un arma con más Covert parecería cara por el arma. Dentro de cada rareza ese efecto prestado desaparece |
| Movimiento 2 | **Compararlo contra el azar.** Con k grupos, η² da en promedio (k−1)/(n−1) aunque los grupos sean arbitrarios, y con 34 armas eso pesa. En las seis rarezas el arma explica entre **2 y 4,9 veces** lo que da el azar, y ninguna queda pegada (el umbral era menos de 1,5 veces) |
| Gráfico | Barras de η² por rareza, con la marca del azar |
| Decisión | **Confirmada:** `weapon` entra (one-hot) |

**Para la defensa:** es un verde ajustado. 0,299 está cerca de 0,25, y en Consumer, Mil-Spec y Covert da amarillo. Las 5 armas más caras son AK-47, AWP, M4A4, M4A1-S y Desert Eagle.

### H2 · Ser souvenir tiene prima de precio — dominio — **VERDE**

| Campo | Contenido |
|---|---|
| Afirmación | La versión souvenir de un skin está más arriba que la normal, porque sólo sale de paquetes de torneo y hay menos |
| Qué esperaba ver | Percentil claramente mayor **comparando el mismo skin en el mismo exterior** |
| Plantilla | Comparación de dos grupos, **en pares** (*d* sobre las diferencias de cada par) |
| Medida | 55 pares, todos sin StatTrak; diferencia media +27,8 y desvío 28,6. **d = 27,8 / 28,6 = 0,971** |
| Zona | **Verde** (> 0,8) |
| Movimientos | **Ninguno:** el pareo ya deja fijo el diseño. Ese era justamente el arreglo pendiente de la versión anterior |
| Gráfico | Histograma de las diferencias por par |
| Decisión | **Confirmada:** `souvenir` entra |

**Lectura sin estadística:** en el 84 % de los pares la souvenir es más cara, y en la mediana vale **12,9 veces** más.

**Si salían menos de 20 pares** (como Steam ordena por nombre, las "Souvenir…" podían quedar lejos de su versión normal en la muestra), el notebook corría una hipótesis de reserva: ¿el mercado paga exactamente la escasez? Salieron 55 y no hizo falta.

### H3 · El exterior mueve el precio del mismo skin — columna del modelo — **VERDE**

| Campo | Contenido |
|---|---|
| Afirmación | Dentro de un mismo skin, más gastado quiere decir más abajo en el mercado |
| Qué esperaba ver | Correlación fuerte y negativa **dentro de cada skin**, mucho más clara que en general |
| Plantilla | Asociación (promedio de los Spearman dentro de cada skin) |
| Medida | **−0,700**, sobre 209 skins con al menos 3 exteriores, sin StatTrak ni Souvenir para aislar el exterior |
| Zona | **Verde** (\|−0,700\| > 0,6) |
| Movimiento | **Medir dentro de cada skin en vez de entre skins.** Entre skins distintos el η² daba **0,051**, amarillo raspando el rojo: un Battle-Scarred de un Covert vale más que un Factory New de un Consumer, así que el exterior quedaba tapado |
| Gráfico | Boxplot por exterior (entre skins) al lado del histograma de las correlaciones dentro de cada skin |
| Decisión | **Confirmada:** `wear_orden` entra |

**Por qué se dan dos números:** cada correlación usa sólo 3 a 5 puntos, así que una sola es ruidosa. Por eso, además del promedio, se da uno que se entiende sin estadística: en el **92 %** de los skins, más gastado es más barato.

### H4 · La probabilidad de drop aporta más allá de la rareza — columna del modelo — **AMARILLO, inconclusa**

| Campo | Contenido |
|---|---|
| Afirmación | `neg_log_p_item` explica la posición por encima de la rareza |
| Qué esperaba ver | **Que no.** `p_item` se calcula con la rareza, el exterior, el StatTrak y el tamaño de la colección, información que ya está en otras columnas. Era la candidata a refutada |
| Plantilla | Asociación controlando la rareza |
| Medida | **0,225** (mediana de \|r\| dentro de las rarezas). Por rareza: Mil-Spec −0,03, Restricted 0,24, Classified 0,22 y Covert 0,44 |
| Zona | **Amarillo** |
| Movimiento | **Medir dentro de cada rareza.** En general daba **0,624, que sería verde**, pero es la rareza disfrazada: la columna se construye con la rareza |
| Gráfico | Dispersión coloreada por rareza, con una recta por color. Si la columna aportara algo propio, cada recta tendría pendiente |
| Decisión | **Inconclusa:** aporta algo, pero poco. Entra como secundaria, con imputación, y se resuelve en la Entrega 3 |

**¿Por qué "inconclusa" y no "refutada"?** Porque 0,225 superó el corte de 0,2, y la regla escrita dice amarillo = inconclusa.

**¿Por qué H1 dio verde y H4 amarillo, si las dos se midieron dentro de la rareza?** Porque el arma tiene información propia: la demanda por la AK no depende de la rareza. `neg_log_p_item`, en cambio, está fabricada casi toda con la rareza.

## 2.6 · La tabla de columnas candidatas

Lleva las cinco columnas que pide el instructivo. La última es el chequeo de fuga.

| Columna | Qué mide | Zona | Decisión | ¿Existiría al predecir? |
|---|---|---|---|---|
| `rarity_orden` | Peldaño de rareza (0 = Consumer … 5 = Covert) | verde (η² 0,408) | Entra **como número**: cada peldaño es 5 veces más raro, así que en log los escalones son iguales. No es una numeración arbitraria | Sí, Valve la publica al lanzar |
| `weapon` | Qué arma es | verde · H1 | Entra (one-hot) | Sí |
| `wear_orden` | Exterior, de Factory New (0) a Battle-Scarred (4) | verde · H3 | Entra | Sí |
| `stattrak` | Si cuenta las bajas | verde (*d* pareada 0,838, 79 pares) | Entra. No tenía ficha, así que se le calculó una medida de respaldo con el método de H2 | Sí |
| `souvenir` | Si es de un paquete de torneo | verde · H2 | Entra | Sí |
| `origen` | Caja, paquete souvenir, drop de mapa o sin identificar | amarillo (η² 0,123) | Entra (one-hot), en duda para la Entrega 3 | Sí |
| `neg_log_p_item` | Qué tan improbable es sacar ese ítem de su caja | amarillo · H4 | Secundaria, con imputación | Sí |
| `antiguedad_col_dias` | Días desde el lanzamiento de la colección | — (dominio) | **Sale:** para un skin nuevo vale ~0, fuera del rango del entrenamiento (3,5 a 13 años) | Sí, pero vale ~0 |
| `caja_discontinuada` | Si la caja salió del pool | — | **Sale:** es la antigüedad disfrazada (las 25 discontinuadas son exactamente las 25 más viejas) | Sí |
| `p_wear` | Fracción del rango de float que cae en el exterior | — (supuesto) | **Sale:** el supuesto quedó parcialmente validado | Sí |
| `category_name` | Rifles, Pistols, SMGs o Heavy | — | **Sale:** redundante con `weapon` | Sí |
| `pattern` | Nombre del diseño | — | **Sale:** casi única por fila, el modelo la memorizaría | Sí |
| `sell_listings` | Publicaciones a la venta | "verde" (r −0,77) | **SALE: FUGA** | **No:** se cuenta junto al precio |
| `sell_price_usd` | Precio | — | **Sale:** de ahí sale el objetivo | No |
| `collection`, `skin_id` | Colección y skin base | — | No son columnas del modelo: agrupan la validación | Sí |

Para la defensa, una distinción:

- **`sell_listings` sale por FUGA:** no existe el día del lanzamiento.
- **`antiguedad_col_dias` sale por DOMINIO:** existe, pero vale un número que el entrenamiento nunca vio.

## 2.7 · Lo que hay que saber explicar (Entrega 2)

Las respuestas están en la celda 8.2 del notebook.

**Cuál es la variable objetivo y cómo está distribuida.**
`percentil_precio`, de 0 a 100 dentro del día. Está repartida pareja por construcción: ~10 % de las filas en cada decil.

**Por qué hay nulos donde hay nulos.**
Los de `neg_log_p_item` y las columnas de caja son **estructurales** (52 % de las filas no sale de una caja). Los de `collection_release` son **faltantes reales** (0,4 %). No se borran: van con imputación más una columna indicadora.

**Qué hipótesis salió mal y qué hicimos.**

- **En la versión anterior:** las cuatro daban amarillo porque comparaban skins distintos. Lo que hicimos fue compararlas en pares, y así H2 y H3 pasaron a verde.
- **En esta versión:** H4 esperábamos refutarla y quedó inconclusa (0,225). La dejamos como secundaria y la resolvimos en la Entrega 3: aporta 0,02 de R².

**Qué columna se descartó por fuga y cuándo se completa en la realidad.**
`sell_listings`. Se completa cuando el skin ya está a la venta: se cuenta en el mismo momento que el precio y cambia en respuesta a él. El día del lanzamiento no existe.

**Qué quedó inconcluso y qué haría falta.**

| Qué quedó inconcluso | Qué haría falta para resolverlo |
|---|---|
| `antiguedad_col_dias` salió por argumento | Medirla con una validación en el tiempo. Se hizo en la Entrega 3 |
| La traducción a dólares es imprecisa en la zona cara (p90: entre 27 y 2.017 USD con ±17 puntos) | Un modelo con menor error |
| La estabilidad entre fechas se midió con sólo 2 snapshots | Más snapshots. En la Entrega 3 se sumó uno: Spearman 0,998 entre el 15/10 y el 30/10 |

## 2.8 · El puente a la Entrega 3 (sección 9)

- **La prueba:** las colecciones más nuevas hasta juntar el **20 % de las filas**. El corte de "menos de un año" daba sólo 5 colecciones, y con tan pocas una colección rara mueve todo el número.
- **El percentil del skin nuevo se mide contra el mercado del entrenamiento.** Si se calculara con los precios de la prueba incluidos, esos precios se filtrarían en las etiquetas del entrenamiento.
- **Las líneas de base** (`GroupKFold` por colección), en MAE: mediana global 25,1, por rareza 18,0, y **por rareza y exterior 17,0**. Ese 17 es el error que se usa en 2.3.

## 2.9 · Cómo se aprueba (Entrega 2)

| Requisito del instructivo | Dónde está |
|---|---|
| El notebook corre entero y tiene las salidas guardadas | Sí, ejecutado con salidas |
| El dataset se describe con números, no con adjetivos | 2.2 y 2.3 |
| Cuatro fichas completas, cada una termina en una decisión | 2.5 |
| La tabla de candidatas existe y pasó el chequeo de fuga | 2.6, última columna |
| Que no conteste uno solo | — |

**Las tres formas de perder los 20 minutos, evitadas:**

- **«EDA de checklist»:** cada sección termina en una decisión.
- **«Contestar con el gráfico»:** cada ficha tiene su número y su zona.
- **«Que hable uno solo»:** depende de nosotros.

---

# ENTREGA 3 · Modelado (`notebooks/Entrega3_modelado_30-10_R2.ipynb`)

## 3.1 · Qué pide el instructivo

> *«¿Tenés un modelo que dice algo que los datos solos no decían, y podés explicar por qué es ese y no otro?»* y *«Un modelo modesto, bien medido y bien explicado vale más que uno brillante que nadie sabe de dónde salió.»*

Nuestro proyecto **tiene objetivo y es un número**, así que es **regresión** y corresponde la sección "si tu proyecto tiene objetivo".

**Qué hay que traer:**

- El notebook ejecutado, con una **semilla definida al principio**: `SEMILLA = 42`, pasada a los modelos, a `KFold` y a la permutación.
- El planteo y el resultado.
- Lo que nos marcaron en la Entrega 2, resuelto.

**Los 25 minutos:**

| Min. | Qué |
|---|---|
| 0–2 | Qué cambió (3.2) |
| 2–7 | El planteo: qué predecimos, la métrica, el piso y la validación (3.3) |
| 7–16 | La comparación, el ajuste y el modelo final: dónde se equivoca y en qué se apoya (3.4 a 3.9) |
| 16–22 | Preguntas dirigidas (3.10) |

## 3.2 · Qué cambió desde la Entrega 2 (dos oraciones)

«El objetivo y las columnas no cambiaron: siguen el percentil y las siete candidatas, y `neg_log_p_item` (H4, amarilla) quedó porque aporta por más que el ruido. Cambiaron los datos (el 30/10 tiene 10.003 filas, cinco veces más), la métrica de elección (R² en vez de MAE, por la corrección de la cátedra) y la validación (en el tiempo, en vez de `GroupKFold` sin orden).»

**Lo que marcaron en la Entrega 2 y cómo se resolvió:**

| Observación | Cómo se resolvió |
|---|---|
| El objetivo eran bandas de precio | Percentil del precio dentro del día, con regresión |
| Las hipótesis comparaban skins distintos y daban amarillo | Comparaciones en pares: H2 y H3 en verde |
| La asimetría se medía sólo en el precio | Se mide en todas las numéricas |
| La tabla de candidatas no decía qué mide cada columna ni su zona | Ahora tiene *Qué mide* y *Zona* |
| H4 no tenía gráfico, y H1 no se comparaba contra el azar | H4 tiene su dispersión por rareza; H1 se compara contra (k−1)/(n−1) |
| El JSON de calidad guardaba la clave `origen` dos veces | Ahora son `procedencia` y `origen_items`; se sumó `max_active_runs=1` |
| El JSON del 15/10 había salido de la rama de respaldo | Se volvió a correr el pipeline el 30/10: `datos_frescos: true`, 10.003 filas |
| «Con un árbol se pueden generar muchas columnas; elijan bien la columna representativa, que no sea el precio» | Sección 8.2 (ver 3.7) |
| Corrección: «elegir con R² en vez de MAE, y quedarse con el mejor R²» | El R² decide todo: modelos, columnas, grilla (`refit="r2"`) y permutación |
| Corrección: «al pasar a dólares, mostrar el error entre el predicho y el real» | Sección 10.3 |

## 3.3 · Las cuatro decisiones previas que pide el instructivo

### 1 · La métrica, declarada antes de elegir: R²

El instructivo dice que en regresión el MAE «es el más fácil de defender». Nosotros empezamos con MAE, y **la cátedra nos corrigió a R²**. Cómo se defiende el R² en este problema:

- **Responde la pregunta tal cual:** ¿qué parte de la variación de la posición de un skin explican sus atributos?
- **Tiene una escala fija:** el percentil está repartido parejo por construcción, así que su varianza es casi igual en todos los pedazos. El R² se compara entre pedazos, entre modelos y contra el piso.
- **Pesa más los errores grandes**, que acá son los caros: poner un skin barato entre los caros.
- **También castiga el sesgo:** un error siempre hacia el mismo lado suma al error cuadrático.
- **Lo que no dice es para qué lado se equivoca.** Por eso va siempre con el **MAE** («le erramos por X puestos de 100») y el **sesgo** («positivo = lo pone más arriba»), como lectura que no decide nada.

### 2 · El piso, medido con la misma validación

- **`DummyRegressor(strategy="mean")`:** R² **−0,094**. Es negativo porque el promedio sale de colecciones viejas y se mide sobre nuevas, que tienen otro promedio: es el primer aviso del corrimiento. Se usa el promedio, no la mediana, porque minimiza el error cuadrático, que es con lo que se arma el R².
- **Pisos informados, una tabla dinámica:** promedio por rareza, R² 0,282, y por rareza y exterior, **0,348**. Si un modelo no les gana, no aprendió nada que una planilla no sepa.

### 3 · La validación imita el uso

La tabla del instructivo tiene tres casos. **Nuestras filas cumplen dos a la vez:**

- **Se repiten por entidad:** el mismo skin genera varias filas hermanas, y una colección comparte época y diseño. El caso del instructivo es `GroupKFold`.
- **Están en el tiempo, y vamos a predecir lo que viene:** el caso del instructivo es `TimeSeriesSplit`.

**Por eso combinamos los dos: pedazos temporales de a colecciones enteras.**

- **La prueba:** las 16 colecciones más nuevas, 2.075 filas (21 %), lanzadas desde el 07/02/2024. Se fijó antes de entrenar y no se cambia después de ver el resultado.
- **La validación cruzada:** se ordenan las 77 colecciones del entrenamiento por fecha. El 40 % más viejo es el punto de partida y el resto va en 4 bloques consecutivos. En cada pedazo se entrena con todo lo anterior y se valida con el bloque siguiente.
- **Lo que se controla con `assert`:** ninguna colección queda de los dos lados, y siempre se valida con colecciones posteriores.

| Pedazo | Entrena (filas) | Valida (filas) | Colecciones que valida, lanzadas entre |
|---|---|---|---|
| 1 | 3.088 | 1.240 | 2015-09 y 2017-04 |
| 2 | 4.328 | 1.185 | 2017-09 y 2019-11 |
| 3 | 5.513 | 1.183 | 2019-11 y 2021-05 |
| 4 | 6.696 | 1.232 | 2021-09 y 2023-04 |

### 4 · Todo adentro de un pipeline

- **Lo que aprende de los datos va adentro**, para que se recalcule en cada pedazo sólo con sus filas de entrenamiento:
  - imputar con la mediana y escalar `rarity_orden` y `wear_orden`;
  - imputar `neg_log_p_item` con la mediana **más la columna indicadora** y escalarla;
  - armar el one-hot de `weapon` y `origen` (un arma que no se vio queda en ceros).
- **El recorte de la predicción a [0, 100] también va adentro**, con un `TransformedTargetRegressor`.
- **Lo único que va afuera** es pasar los booleanos a 0/1: es una conversión de tipo que no aprende nada.
- **Es el mismo preprocesamiento para todos los modelos**, así la comparación es justa.

## 3.4 · La comparación

**Qué pide el instructivo:** el modelo simple primero, al menos dos familias más con media y desvío, y «si el complejo no le gana al simple por más que el desvío, se queda el simple».

| Modelo | R² (media ± desvío) | MAE | Sesgo |
|---|---|---|---|
| **Gradient Boosting** (`HistGradientBoostingRegressor`) | **0,602 ± 0,093** | 14,8 | +8,6 |
| Regresión lineal (Ridge, la referencia) | 0,560 ± 0,111 | 15,7 | +9,2 |
| Random Forest | 0,535 ± 0,087 | 15,4 | +8,5 |
| Árbol de decisión | 0,417 ± 0,071 | 16,7 | +8,4 |
| Piso: rareza y exterior | 0,348 ± 0,069 | 19,5 | +9,3 |
| Piso: promedio | −0,094 ± 0,109 | 26,4 | +6,7 |

**La regla, escrita antes de ver la tabla:** se queda el de mayor R²; si no es la lineal, tiene que ganarle a la lineal por más que el ruido.

**Qué "desvío" se usa, y por qué en pares.** El desvío de cada modelo (±0,09, ±0,11) mide sobre todo qué tan difícil es cada pedazo, y eso lo comparten todos los modelos. Como todos se miden sobre los **mismos** pedazos, se resta pedazo a pedazo y se toma el desvío de esa diferencia. Lo difícil de cada pedazo se cancela. Es la misma lección de H2 y H3.

| Contra la lineal | R² que suma | Desvío de la diferencia | Gana en | ¿Supera el ruido? |
|---|---|---|---|---|
| **Gradient Boosting** | **+0,042** | **0,020** | **4 de 4** | **Sí → elegido** |
| Random Forest | −0,024 | 0,057 | 2 de 4 | No |
| Árbol de decisión | −0,142 | 0,059 | 0 de 4 | No |

- **Por qué Gradient Boosting puede ganarle a la lineal:** la lineal suma un efecto por columna; los árboles aprenden **combinaciones**, como que el origen pese distinto según la rareza o que "souvenir sin identificar" signifique escasez.
- **Por qué con el 15/10 ganaba la lineal:** con 2.012 filas la ventaja no superaba el ruido. Con cinco veces más filas, sí.
- **Por qué Ridge y no una lineal pura:** con 34 armas en one-hot, sin regularizar los coeficientes quedan inestables.

**¿Y si los pedazos se arman de otra forma?** El instructivo pregunta qué le pasaría al número con pedazos al azar:

| Esquema | R² | Sesgo |
|---|---|---|
| Al azar (KFold) | **0,670** (desvío 0,008) | −0,0 |
| Por skin base | 0,606 | +0,1 |
| Por colección, sin orden | 0,581 | +0,0 |
| **Por colección y en el tiempo (el nuestro)** | 0,602 | **+8,6** |

- **Al azar, el R² sale 0,068 más alto y casi sin desvío.** Es «el número lindo que no da ningún aviso» del instructivo.
- **Lo importante:** el R² solo no alcanza para ver el problema (por colección sin orden da incluso menos). Lo que separa a la validación temporal es el **sesgo**, +8,6 contra ~0. Las otras tres mezclan épocas y el corrimiento se promedia y desaparece.

## 3.5 · El ajuste de hiperparámetros del elegido

**Qué pide el instructivo:** una grilla, con la tabla completa y no sólo el ganador.

- **La grilla:** 24 combinaciones (`learning_rate` 0,03 / 0,1, `max_leaf_nodes` 15 / 31, `min_samples_leaf` 10 / 20 / 40 y `loss` squared / absolute), con los mismos 4 pedazos y `refit="r2"`. El notebook muestra la tabla completa.
- **El ganador:** `learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=40` y `loss="squared_error"`.
- **Cuánto mejora:** el R² pasa de 0,602 a **0,614 ± 0,103**, con una diferencia contra los valores por defecto de +0,013 ± 0,012. **Supera el ruido, pero por muy poco**, y se presenta como una mejora chica.
- **Por qué `squared_error`:** el R² está hecho con el error cuadrático, así que conviene entrenar minimizándolo. Con el MAE, la grilla elegía `absolute_error`. **La métrica de elección decide también la pérdida.**

**Clases desparejas:** no aplica. El objetivo es un número, y además es parejo por construcción.

## 3.6 · Las columnas que probamos y no sirvieron, con números

**Qué pide el instructivo:** «también son evidencia».

**Una columna por vez** (cuánto baja el R² al sacarla):

| Columna | Baja | Lectura |
|---|---|---|
| `rarity_orden` | 0,145 | aporta |
| `wear_orden` | 0,059 | aporta |
| `origen` | 0,059 | aporta |
| `neg_log_p_item` | 0,020 ± 0,008 | aporta |
| `stattrak` | 0,011 ± 0,006 | aporta |
| `souvenir` | −0,001 | dentro del ruido |
| `weapon` | −0,019 ± 0,022 | dentro del ruido |

**Las dos columnas en duda (`origen` y `neg_log_p_item`), en las cuatro combinaciones.**

- **Por qué no alcanza con sacarlas de a una:** comparten información. `neg_log_p_item` tiene valor **exactamente** cuando `origen` es "caja" (lo muestra un cruce de tablas), así que su columna indicadora es una copia de una columna del one-hot de `origen`. Si dos columnas se repiten, sacar cualquiera sola no cambia nada, y eso no prueba que se puedan sacar las dos.
- **La regla, escrita antes de medir:** una combinación es aceptable si no baja el R² por más que el desvío, y de las aceptables se queda la que tiene menos columnas.
- **Resultado:** sin `neg_log_p_item` baja 0,020, sin `origen` baja 0,059 y sin las dos, **0,185**. Ninguna con menos columnas es aceptable, así que **quedan las 7**.
- **Así se cierra H4:** la probabilidad de drop aporta, poco.

**`souvenir` y `weapon` se quedan aunque estén dentro del ruido.** No prueban que se puedan sacar, y tienen respaldo verde: H2 en pares y H1 por rareza. Con los hiperparámetros ajustados, `weapon` se volvió a medir: +0,004 ± 0,019, sigue dentro del ruido.

**`antiguedad_col_dias`, medida.** Con ella el R² **se derrumba** de 0,602 a 0,227 (con desvío 0,603) y el sesgo empeora de +8,6 a +13,0. Un árbol no puede extrapolar a 0 días, una edad que nunca vio. **No entra**, y queda medido el argumento de la Entrega 2.

**Lo que marcó el profesor, la sección 8.2.**

- Con one-hot, las 7 columnas se vuelven **44**. Se probó la alternativa de **una sola columna por categórica** (`OrdinalEncoder`, 8 columnas) en las tres familias de árboles, y **ninguna le ganó al elegido**.
- Un árbol chico muestra que su **primera pregunta es `rarity_orden`**, y después `origen`.
- Un `assert` garantiza que ninguna columna del árbol sale del precio.

## 3.7 · El modelo final, leído

### Medido una vez sobre el test

Antes de mirar se decidió que, pase lo que pase, la aplicación usa el mismo modelo **reentrenado con todas las filas**. Así el resultado del test no puede influir en la elección.

| | R² | MAE | Sesgo |
|---|---|---|---|
| Piso: promedio | −0,149 | 31,9 | +12,7 |
| **Piso: rareza y exterior** | **0,308** | 23,7 | +11,3 |
| **Modelo final** | **0,272** | 23,4 | **+22,6** |

**El resultado principal:** el R² pasa de 0,614 en validación a **0,272 en la prueba**, y en la prueba **queda debajo de la tabla de rareza y exterior**. El instructivo lo contempla: «si tu modelo no le ganó al piso, vení igual y decilo».

**Por qué pasa, explicado desde el dominio:**

- **No es una colección rara:** el sesgo es positivo en **las 16**.
- **No es una columna rota:** hasta el piso, que no mira columnas, se pasa por +12,7. La mediana del percentil en la prueba es 27, contra 50 en el entrenamiento.
- **Causa 1, la oferta:** una colección recién lanzada se sigue consiguiendo, así que tiene mucha oferta. Las viejas salen del pool.
- **Causa 2, el Armory** (desde octubre de 2024): **"drop de mapa" cambió de significado.**
  - En el entrenamiento son 38 colecciones de mapa viejas, que ya no caen y son caras: percentil mediano **50,9**.
  - En la prueba son 13 colecciones, casi todas del Armory, un pase que se compra: abundan y son baratas. Percentil mediano **9,3**, con un sesgo de **+31,6**.
  - Entre la última colección de mapa del entrenamiento (abril de 2023) y la primera de la prueba (octubre de 2024) no hay ninguna, así que el modelo no tenía cómo ver el cambio.
- **Qué se hace:** no se corrige mirando la prueba, porque eso la convertiría en entrenamiento. Se declara, y el modelo de la aplicación se reentrena con todas las colecciones, **incluidas las del Armory**.

### Dónde se equivoca

El instructivo pide, en regresión, lo real contra lo predicho y en qué tramo se erra más:

- **Real contra predicho:** está en la sección 10.1.
- **Por tramo:** el error se concentra en los baratos. En 0–10 el MAE es 33 y el sesgo +33; en 90–100, el MAE es 4,8. Acá se usa el MAE y no el R², porque dentro de un tramo de 10 puntos el percentil casi no varía y el R² dividiría por casi nada.
- **En dólares** (la corrección de la cátedra):
  - el error típico es de **7,50 USD** (mediana), y el promedio sube a 39,25 USD por los skins caros;
  - el error relativo mediano es de **562 %**, y sólo el 20 % de los skins queda entre la mitad y el doble de su precio;
  - el **92 %** de los predichos es más caro que el real;
  - en el último tramo el signo se da vuelta (−168 USD): Gradient Boosting promedia hojas y no llega a los extremos.
- **Los errores más grandes:** los diez son "drop de mapa" de colecciones recientes (Boreal, Radiant, Dead Hand…), puestos entre 57 y 60 puntos más arriba. El error al revés es el Souvenir MAC-10 | Bronzer: vale 500 USD y el modelo predice 30, porque su precio lo fija una escasez puntual que no está en ninguna columna.

### En qué se apoya

El instructivo pide los coeficientes con su signo **o** la permutación. Están las dos.

- **Permutación** (en la prueba, en R²): `rarity_orden` 0,68, `origen` 0,51, `wear_orden` 0,08, `neg_log_p_item` 0,07, `weapon` 0,01, `souvenir` 0,01 y `stattrak` ~0.
- **Coeficientes de la lineal de referencia:** el de `stattrak` sale **negativo**, al revés que en la Entrega 2. No es una contradicción: StatTrak existe casi sólo en ítems de caja (2.475 contra 135 de mapa), y en el mercado establecido los de caja están más abajo. La lineal no separa las dos cosas. **En pares** (2.088 pares del mismo skin y exterior), la versión StatTrak está más arriba en el **83 %**.
- **«Si la columna que más pesa te sorprende, sospechá de una fuga»:** la que más pesa es la rareza, la esperable, que existe el día del lanzamiento y no sale del precio. Además, el R² de la prueba (0,27) no es sospechosamente alto.

## 3.8 · Lo que se pregunta (Entrega 3)

Las respuestas están en la sección 11 del notebook.

**1. Por qué esa métrica.**
El R² responde «cuánto de la posición se explica con los atributos», tiene una escala fija porque el percentil es parejo, pesa los errores grandes y castiga el sesgo. Va acompañado del sesgo porque el error va siempre hacia arriba, y el R² no lo dice.

**2. Cuánto le gana al piso, y si es mucho o poco.**

| | Piso | Modelo | Diferencia |
|---|---|---|---|
| Validación, contra el promedio | −0,094 | 0,614 | +0,708 |
| Validación, contra la tabla de rareza y exterior | 0,348 | 0,614 | +0,267 |
| Prueba, contra la tabla de rareza y exterior | 0,308 | 0,272 | **pierde** |

En el mercado establecido, explicar ~60 % sin ver el diseño alcanza para decir si un skin va a quedar entre los baratos, los del medio o los caros. Para las colecciones nuevas no alcanza.

**3. Por qué los pedazos se arman así.**
Por colección, porque hay filas hermanas, y en el tiempo, porque el modelo es para colecciones futuras. Al azar, el R² saldría 0,068 más alto y el sesgo de +8,6 desaparecería.

**4. Por qué ese modelo y no uno más complejo.**
Gradient Boosting le gana a la lineal por 0,042 ± 0,020, en los 4 pedazos. Random Forest y el árbol quedan debajo de la lineal.

**5. Un tramo donde erra mucho, explicado desde el dominio.**
El tramo 0–10 (MAE 33): skins baratos del Armory que el modelo pone como si fueran de mapa viejo.

**6. La columna que más pesa y por qué no es fuga.**
`rarity_orden`: es la esperable, existe al lanzar y no se calcula con el precio. `sell_listings`, la única atada al precio, quedó afuera desde la Entrega 2.

**Lo que probamos y no anduvo** (el instructivo lo pide explícitamente):

- la regresión lineal, le ganó Gradient Boosting;
- `antiguedad_col_dias`, el R² se derrumba;
- sacar `weapon`, da igual (dentro del ruido);
- una columna por categórica en los árboles, ninguna variante gana;
- en el ajuste, la pérdida `absolute_error`, pierde con el R².

## 3.9 · Cómo se aprueba (Entrega 3)

| Requisito del instructivo | Dónde está |
|---|---|
| Corre entero, con semilla y salidas | `SEMILLA = 42` y scikit-learn fijado en 1.6.1 |
| Todo adentro de un pipeline | 3.3 punto 4 |
| La métrica declarada | 3.3 punto 1 |
| El piso medido | 3.3 punto 2 |
| Al menos 3 modelos con media y desvío | Tenemos 4 |
| El elegido, ajustado | 3.5 |
| El test, mirado una vez | 3.7 |
| El modelo final, leído | 3.7 |
| Saber decir qué se probó y no anduvo | 3.8 |

**Las formas de perder los 25 minutos, evitadas:**

- **«El número que da demasiado bien»:** se midió explícitamente el optimismo de la validación al azar.
- **«El zoológico»:** son 4 familias con desvío, elegidas por validación cruzada y no por el test.

---

# ENTREGA 4 · Visualización y aplicación (`build_app.py` + `app/plantilla.html` → `docs/index.html`)

## 4.1 · Qué pide el instructivo

> *«¿Alguien que no estuvo en tu proyecto puede entender lo que encontraste, creerte, y después hacerle sus propias preguntas a tus datos?»*

Es la **versión final**: una aplicación en una URL pública, la misma que se muestra en la exposición del 25/11.

**Qué hay que traer:**

- La URL pública ya abierta: https://cs2skins-docs.vercel.app/ («que funcione en la notebook de alguien no alcanza»).
- El repositorio en GitHub con el README.
- Lo que nos marcaron en la Entrega 3, resuelto.

**Los 25 minutos:**

| Min. | Qué |
|---|---|
| 0–2 | Qué se corrigió del modelo desde la Entrega 3, en dos oraciones (4.2) |
| 2–10 | El tallo: de la portada al modelo, en orden, **como si el docente no conociera el proyecto**. El docente abre los «Cómo lo sabemos» que quiera |
| 10–15 | La copa: **el docente toma el mouse**, explora y prueba un caso propio |
| 15–22 | Preguntas dirigidas sobre la aplicación o el repositorio (4.9) |

## 4.2 · Qué cambió desde la Entrega 3 (dos oraciones)

«El modelo es el mismo de la Entrega 3 con las dos correcciones aplicadas: se elige con R² en vez de MAE (la grilla pasó a `squared_error`) y el error en dólares se muestra como precio predicho contra precio real. La aplicación lo reproduce con la misma semilla, comprueba que da los mismos números que el notebook y usa la versión reentrenada con todas las filas, incluidas las colecciones del Armory.»

*(Si la devolución de la Entrega 3 marcó otra cosa además de estas dos correcciones, sumarla acá. En el notebook sólo están documentadas estas dos.)*

## 4.3 · La copa de martini: qué pide cada sección y dónde está

| Parte | Sección que pide | Nuestra sección | Cierra |
|---|---|---|---|
| Portada | El proyecto | «¿Se puede saber cuánto va a valer un skin antes de que salga a la venta?» | La definición |
| — | (agregado nuestro) | «CS2 en dos minutos»: glosario del juego para quien no juega | — |
| Tallo | Los datos | «La mitad de los skins cuesta menos de USD 4,27, pero uno de cada diez cuesta más de…» | Entrega 1 |
| Tallo | Lo que encontramos | «La rareza ordena el precio, siempre que compares skins que se consiguen de la misma forma» | Entrega 2 |
| Tallo | El modelo | «El modelo aprendió bien el mercado viejo, y justamente por eso se equivoca con el Armory» | Entrega 3 |
| Copa | Explorá | «Los 10.003 skins en una sola imagen» | Todas |
| Copa | Probá el modelo | «Fabricá un skin y mirá en qué puesto lo pone el modelo» | Entrega 3 |

## 4.4 · La portada

**Qué pide el instructivo:** la pregunta, **la idea central en una oración con sujeto y verbo**, y **tres números que la sostienen**.

- **La pregunta:** «¿Se puede saber cuánto va a valer un skin antes de que salga a la venta?»
- **La oración:** «En parte: con los datos del lanzamiento, **el modelo explica el 61 %** del orden de precios en colecciones que ya conocía, pero con las 16 más nuevas **cae a 0,27** y pone más caros de lo que valen al **92 %** de sus skins, porque desde octubre de 2024 el Armory las abarató.»
- **Los tres números:**
  - **0,614**: el R² con colecciones viejas que no vio (la tabla simple saca 0,35);
  - **0,272**: el R² en la prueba (la tabla saca 0,31, y ahí el modelo no le gana);
  - **92 %**: los skins de la prueba que el modelo puso más caros de lo que valen.
- **Un gráfico de entrada:** cada raya es un skin, ubicado por su puesto y con una fila por rareza. El título afirma: «Cuanto más rara es una skin, más a la derecha: más cara».

**Por qué la oración dice "en parte".** Es la misma conclusión que defendimos en la Entrega 3: el modelo funciona con el mercado conocido y se cae con el nuevo. El instructivo lo permite: «una historia honesta sobre un resultado modesto vale más que una aplicación brillante que no se sabe de dónde salió».

## 4.5 · El tallo: cada sección, un hallazgo

El instructivo pide que cada sección tenga tres partes:

1. **Un título que afirma** (no «Distribución de X», sino algo que se pueda discutir).
2. **Un gráfico que prueba el título**, con unas pocas líneas que dicen qué mirar.
3. **«Cómo lo sabemos»**, un desplegable cerrado, con la evidencia técnica.

### Los datos

- **El título que afirma:** «La mitad de los skins cuesta menos de USD 4,27, pero uno de cada diez cuesta más de…».
- **El gráfico:** un histograma del precio con el **eje en escala log** (0,1 → 1 → 10 → 100). Con un eje común, los baratos quedarían aplastados contra el borde. Un segundo gráfico muestra la escalera de rareza dentro de las cajas: cada escalón multiplica el precio entre 4 y 6 veces, y un Covert típico vale 161 veces más que un Mil-Spec.
- **Qué mirar:** con precios tan desparejos, errarle por USD 5 es muchísimo para un skin de USD 0,10 y nada para uno de USD 1.000. Por eso se predice el puesto.
- **«Cómo lo sabemos», con lo que pide el instructivo:**

| Qué pide | Qué hay |
|---|---|
| De dónde salen y cómo se obtienen: el pipeline y la fecha | Airflow con 7 tareas, bronce y plata, corrida del 30/10 con datos frescos |
| Qué es una fila | Un skin de arma en una versión concreta |
| Qué se filtró, paso a paso, con cuántas filas quedan y por qué | La tabla del embudo (1.2) |
| Qué columnas se descartaron por fuga | `sell_price_usd` y `sell_listings`, más `antiguedad_col_dias`, que no es fuga pero tampoco entra |
| (agregado nuestro) Por qué no hay una columna «Armory» | El catálogo no lo dice. En la página se marca por fecha, y el modelo no usa esa marca |

- **Un aviso del Armory** antes de seguir: «Es como estudiar con exámenes viejos y que justo antes del final te cambien el programa».

### Lo que encontramos

- **El título que afirma:** «La rareza ordena el precio, siempre que compares skins que se consiguen de la misma forma».
- **El gráfico:** el puesto mediano por rareza, con una línea por origen. Todas las líneas de color suben. La línea gris, que mezcla todos los orígenes, se quiebra: los Consumer parecen más caros que los Mil-Spec porque caen en colecciones de mapa viejas y escasas. **De ahí que el modelo necesite saber el origen, no sólo la rareza.**
- **Dos gráficos más sobre el Armory:**
  - Las colecciones de mapa por año: desde el Armory ninguna pasa del puesto 37, y un skin típico de mapa pasó de USD 5,36 a **USD 0,27**, 20 veces menos.
  - La misma rareza antes y después: un Mil-Spec de mapa valía USD 13,43 y con el Armory vale USD 0,30, 45 veces menos. Así se descarta que "el Armory traiga rarezas más bajas".
- **«Cómo lo sabemos»**, con lo que pide el instructivo: **una hipótesis que se sostuvo y una que no, cada una con su número**, y cómo se armó el gráfico.

| Hipótesis | Número en la página (30/10) | Resultado |
|---|---|---|
| **H3**, el exterior | Mediana **−0,90** en 1.169 diseños; en el 93 %, más gastado es más barato | Se sostuvo |
| **H4**, la dificultad de drop | **0,199** dentro de cada rareza, sobre 4.948 ítems de caja: rojo | No se sostuvo |
| H2, souvenir | d = 0,92 en 1.631 pares, mediana de +18 puestos | También se sostuvo |

  - **Cómo se construyó el gráfico:** la mediana del puesto por origen y rareza, sólo con grupos de 20 skins o más. Se usa la mediana porque unos pocos skins carísimos tirarían el promedio para arriba.

### El modelo

- **El título que afirma:** «El modelo aprendió bien el mercado viejo, y justamente por eso se equivoca con el Armory».
- **El gráfico principal:** para cada forma de predecir, el R² en validación (círculo lleno) y en la prueba (triángulo hueco). El modelo (ámbar) cae más que la tabla simple (gris).
- **Seis pasos en lenguaje llano:**
  1. qué aprendió;
  2. cómo se puso a prueba;
  3. dónde se equivoca: un real contra predicho donde los skins del Armory, **el 54 % de la prueba**, quedan en un **98 %** arriba de la línea, subidos +32 puestos contra +12 del resto;
  4. por qué: «drop de mapa» cambió de significado;
  5. qué significa en plata;
  6. qué nos llevamos.
- **«Cómo lo sabemos»**, con todo lo que pide el instructivo:

| Qué pide | Qué hay |
|---|---|
| La métrica y por qué | R² + sesgo |
| El piso | −0,09, y la tabla de rareza y exterior con 0,35 |
| Cómo se armó la validación | La tabla de los 4 pedazos |
| La comparación con media y desvío | La tabla y su gráfico de intervalos |
| La regla en pares | La tabla de duelos |
| El ajuste del elegido | La grilla de 24 combinaciones y los ganadores |
| El resultado en el test | La tabla de la prueba |
| Dónde se equivoca | Tabla por tramos, en puntos y en dólares, y los peores errores |
| En qué se apoya | La permutación |
| (agregado nuestro) La reproducción | La frase **«Reproducido por build_app.py… coincide con el notebook: R² 0,614 en validación y 0,272 en la prueba»** |

## 4.6 · La copa: lo que pregunta el usuario

### Explorá

El instructivo pide que vaya de lo general a lo particular:

| Pide | Qué hicimos |
|---|---|
| **Un panorama** | Los **10.003 skins** en una sola nube: x = año de la colección, y = precio en escala log, color = rareza (los colores del juego), con la franja del Armory marcada |
| **Zoom y filtro, al menos dos controles** | **Cuatro:** cómo se consigue (menú), arma (menú), variante (menú) y rareza (chips) |
| **Detalle a pedido, incluido lo que dijo el modelo** | Al tocar un punto se abre la **ficha**: imagen del skin, colección, precio, puesto real, y el puesto y precio que estimó el modelo |

**Por qué la estimación de la ficha nunca viene de un modelo que haya visto ese skin.**

- Los de las 16 colecciones nuevas usan **el modelo evaluado** del notebook.
- Los demás usan el mismo modelo, con los mismos hiperparámetros, **entrenado sin la colección del skin**: una validación cruzada por colección en 5 grupos, con R² 0,60.
- Si se usara el modelo entrenado con todo, la página mostraría predicciones sobre skins ya vistos y el modelo parecería mejor de lo que es. Es la misma lógica anti-fuga de la Entrega 3.

### Probá el modelo («Fabricá un skin»)

- **Qué carga el usuario:** arma, rareza, exterior, cómo se consigue, variante y dificultad de drop.
- **Qué devuelve:** el **puesto predicho**, su **precio en el mercado de hoy** y un **rango de ± 14,6 puntos** (el MAE en validación), dibujado como una banda sobre los 10.003 skins reales.
- **Lo que pide el instructivo:** «el usuario no puede cargar valores que el modelo nunca vio, o la aplicación le avisa cuando lo hace». **Hicimos las dos cosas:**
  - **No se puede cargar lo imposible:** las variantes dependen del origen, como en el juego. Caja y mapa pueden ser Normal o StatTrak; paquete souvenir y sin identificar, sólo Souvenir. La dificultad de drop sólo aparece para ítems de caja.
  - **Avisa cuando algo no se vio:** si la combinación de arma y rareza o de origen y rareza no está en los datos («el modelo nunca vio esta combinación… es una extrapolación»), si hay menos de 50 casos de ese origen y rareza, o si hay pocos StatTrak de ese origen.
- **Agregados:**
  - una **vitrina** con un diseño real que tiene esos atributos y su imagen de Steam. «Ver otro diseño» cambia la imagen, pero **no la predicción**, porque el modelo no mira el diseño;
  - el botón **«Copiar un skin real al azar»**.

## 4.7 · Los gráficos

**Lo que pide el instructivo y cómo se cumple:**

- **Hechos con Altair:** todos se arman en `build_app.py` con Altair y viajan como especificaciones de Vega-Lite (el objeto `ESPECS`).
- **Colores, categorías y escalas con el mismo significado en toda la aplicación:**

| Color | Significa siempre |
|---|---|
| Los 6 colores del juego | La rareza: Consumer `#9fb3cc` … Covert `#eb4b4b` |
| Ámbar | El modelo |
| Salmón | El Armory |
| Gris | Las referencias y los pisos |
| 4 colores propios | Los orígenes |

  **`auditar_colores.py`** recorre la página generada y lista dónde se usa cada color, para comprobar que se respeta **un color = un rol**. El resultado está en `capturas/auditoria_colores.txt`.
- **La copa es interactiva, y cada gráfico del tallo dice lo suyo sin que nadie lo toque:** el tallo tiene tooltips como extra, pero cada gráfico trae su frase de "qué mirar".

### Un gráfico, componente por componente: Explorá

Es la pregunta típica de la defensa.

| Componente | Qué es | Por qué |
|---|---|---|
| **Datos** | Los 10.003 skins, que viajan dentro de la página (`APP.SKINS`) | En HTML no corre Python |
| **Transformaciones** | 1) `transform_filter` con los 4 controles; 2) `transform_calculate` que corre cada punto unos días a izquierda o derecha (un *jitter* de ±50 días, con la semilla) | Los skins de una colección comparten fecha y, sin el jitter, quedarían apilados en una raya vertical. La fecha real está en la ficha |
| **Marca** | `mark_circle`, tamaño 18, opacidad 0,8 | Son 10.000 puntos: tienen que ser chicos y semitransparentes para que se vea la densidad |
| **x** | `fecha:T`, el lanzamiento de la colección | Muestra el tiempo, que es la clave de la historia (viejas contra nuevas, el Armory) |
| **y** | `p:Q`, el precio, en **escala log**, con marcas en 0,1 / 1 / 10 / 100 / 1.000 y dominio fijo | La escala log por la cola larga. El dominio fijo hace que **el eje no salte al filtrar** |
| **Color** | `rareza:N` con la escala de los colores del juego | Es la misma escala que en todo el resto de la página |
| **Orden** | por `rarity_orden` | Los Covert (rojos) se dibujan encima |
| **Capa extra** | Un rectángulo salmón desde el 02/10/2024 | Marca la franja del Armory |

### Qué componente toca cada control

- **En Explorá**, cada menú fija un **parámetro de Vega** (`f_origen`, `f_arma`, `f_var` y `f_rareza`) con `view.signal(...)`. Esos parámetros alimentan el **`transform_filter`**: cambian **qué filas entran al gráfico**. No cambian las escalas ni los ejes, que tienen dominio fijo. Tocar un punto dispara un evento de clic que llena la ficha con el `datum` del punto.
- **En Probá**, al cambiar cualquier atributo se buscan la predicción y su rango, y se fijan las señales `pred`, `lo` y `hi`. Esas señales mueven la **posición x** de la línea ámbar, del triángulo y de la **banda** (`mark_rect`, de `lo` a `hi`). El fondo de rayas de los 10.003 skins no cambia.

## 4.8 · Dónde corre: HTML (GitHub Pages), y cómo llega el modelo

**El problema que plantea el instructivo:** en HTML no corre Python, así que «la fórmula tiene que poder escribirse adentro de la página: una regresión lineal, logística, o un árbol». Nuestro modelo es un **Gradient Boosting**, que no se escribe a mano. El instructivo ofrece dos salidas: **Streamlit**, o un modelo que sí se pueda llevar y quede a menos de un desvío del mejor.

**La salida que tomamos: precalcular todas las respuestas posibles.**

- Todas las entradas del modelo son **discretas**: 34 armas, 6 rarezas, 5 exteriores, 4 orígenes y las variantes que permite cada origen.
- Así que `build_app.py` le pide al modelo real la predicción de **cada combinación posible: 10.200 casos**. Viajan como una tabla dentro de la página, y la página **sólo busca**. **No hay ninguna fórmula aproximada**: es exactamente el Gradient Boosting del notebook.
- **La única simplificación es `neg_log_p_item`**, que es continua. Para los ítems de caja va en **tres niveles**: fácil, típica y difícil, que son los cuantiles 15, 50 y 85 de los ítems reales con la misma rareza, exterior y variante. Si esa combinación tiene menos de 5 ítems, se usan todos los de caja.
- Está declarado en el «Cómo llega el modelo a esta página» de la sección Probá.

**Por qué no las otras dos salidas.**

- **Una lineal o un árbol:** habría que cambiar de modelo. La lineal queda **0,042 por debajo**, y además Random Forest y el árbol tampoco le ganan a la lineal (3.4). Con la tabla precalculada no hace falta resignar nada.
- **Streamlit:** hay que mantenerlo publicado y andando, y se duerme si nadie lo usa. GitHub Pages es una página estática que abre siempre.

**Por qué `build_app.py` reentrena en lugar de cargar el `.joblib`.** Reproduce el notebook paso a paso (partición, pedazos, pisos, las cuatro familias, la grilla, la prueba y la permutación), con la misma semilla y scikit-learn 1.6.1. Al final **compara** su R² de validación, R² de prueba, MAE y sesgo con `data/modelo_metadatos_notebook.json` (tolerancia 0,006) y **corta si no coinciden**. Así la página no puede mostrar algo distinto del notebook, que es uno de los requisitos de aprobación. Después reentrena con todas las filas, como el `modelo_percentil_cs2.joblib` del notebook, para la sección Probá.

## 4.9 · Lo que hay que saber explicar (Entrega 4)

**Por qué cada título del tallo afirma lo que afirma, y qué número lo sostiene.**

| Título | Lo sostiene |
|---|---|
| Los datos: «La mitad cuesta menos de USD 4,27…» | La mediana del precio y el percentil 90 del día |
| Lo que encontramos: «La rareza ordena el precio, siempre que compares skins que se consiguen de la misma forma» | Las líneas por origen suben todas. Dentro de las cajas, cada escalón multiplica el precio entre 4 y 6 veces |
| El modelo: «Aprendió bien el mercado viejo, y por eso se equivoca con el Armory» | R² 0,614 → 0,272. Sesgo +32 en los skins del Armory contra +12 en el resto. 98 % de los del Armory sobreestimados |

**Un gráfico, componente por componente.**
La tabla de Explorá (4.7). Conviene que cada uno sepa leer también otro, por ejemplo el de líneas de «Lo que encontramos»:

- x = rareza (ordinal, de común a rara);
- y = puesto mediano (0 a 100);
- color = origen;
- marca = línea con puntos;
- la línea gris punteada mezcla todos los orígenes.

**Una decisión de diseño que cambiamos.**
Salen de comparar la primera versión publicada (commit del 11/11) con la actual. **Confirmen entre ustedes el motivo de cada cambio antes de la defensa:**

1. **Las barras de importancia eran azules (`#4b69ff`), que es el color de Mil-Spec.** En una página donde el color quiere decir rareza, eso hacía pensar que la importancia tenía que ver con Mil-Spec. Se pasaron al **ámbar del modelo**, y se sumó `auditar_colores.py` para no repetir el error. Es el ejemplo más fácil de defender, porque viola directamente la regla de «un color = un rol».
2. **Explorá** era antes un real contra predicho con un selector de época que había que arrastrar. Se cambió por la nube de todos los skins (año × precio, con los colores de rareza) y cuatro menús. El real contra predicho ya está en el paso 3 del modelo, el panorama tiene que mostrar *todos los datos*, y arrastrar un selector no se descubre solo.
3. **El histograma de precios era dorado** (`#c99a4b`, muy parecido al ámbar del modelo). Pasó a gris, para reservar el ámbar al modelo.
4. **El título del modelo** era «El modelo explica el 61 %… pero en las colecciones nuevas no le gana a una tabla». Pasó a «aprendió bien el mercado viejo, y justamente por eso se equivoca con el Armory»: afirma la **causa**, no sólo el número.

**Qué componente toca cada control.** Está en 4.7.

**Cómo llega el modelo a la aplicación, y qué pasa si el usuario carga un caso raro.**
Llega por la grilla precalculada de 4.8. Ante un caso raro:

- lo imposible no se puede elegir;
- lo que no está en los datos genera un aviso de extrapolación;
- un nivel de dificultad sin datos suficientes usa todos los ítems de caja.

**Cómo se regenera la aplicación si cambian los datos.**
Se corre el pipeline, se copia el CSV nuevo (con su `calidad_*.json`) a `data/`, se corre `python build_app.py --csv data/cs2_skins_dataset_AAAA-MM-DD.csv` (alrededor de un minuto) y se hace commit de `docs/index.html`, que GitHub Pages publica. Con un snapshot de otra fecha, el script avisa que los números son nuevos en lugar de comparar con el notebook. `imagenes.py` se corre una vez para bajar las imágenes.

## 4.10 · Cómo se aprueba (Entrega 4)

| Requisito del instructivo | Estado |
|---|---|
| La aplicación abre en su URL pública, sin errores | **Probarla desde otra computadora antes de entrar** |
| Portada con la oración, y las 3 secciones del tallo con título que afirma, gráfico y «Cómo lo sabemos» | 4.4 y 4.5 |
| La copa: panorama, al menos 2 controles, detalle con el resultado del modelo, probar un caso nuevo | 4.6 |
| Gráficos con Altair, y el grupo puede leerlos componente por componente | 4.7 |
| Coincide con el notebook, y el README permite regenerarla | El `assert` de `build_app.py` y el README |
| Que no conteste uno solo | — |

**Las tres formas de perder los 25 minutos, evitadas:**

- **«El tablero sin historia»:** cada sección del tallo afirma algo, y la copa viene después.
- **«El notebook con otra ropa»:** lo técnico está en los «Cómo lo sabemos», no en el tallo.
- **«La aplicación que sólo corre en una notebook»:** está en GitHub Pages.

---

## 5 · Lo que conviene revisar antes de la defensa

1. **La lectura de «por más que el desvío» (Entrega 3).** El instructivo dice «si el complejo no le gana al simple por más que el desvío, se queda el simple». Si el profe lo lee como el desvío de cada modelo entre pedazos (±0,093 y ±0,111), **la ventaja de Gradient Boosting (0,042) no lo supera**. Nuestra regla usa el desvío de la **diferencia en pares** (0,020). Tener lista la explicación de 3.4:
   - Los dos modelos se midieron sobre los mismos pedazos, así que lo difícil de cada pedazo es común y se cancela al restar.
   - Gradient Boosting gana en los **4 de 4** pedazos.

   Lo mismo pasa con el ajuste: +0,013 ± 0,012 supera el ruido por muy poco, y se presenta así.
2. **H4 cambia de nombre entre entregas.**
   - En la Entrega 2 (15/10) es **«inconclusa, amarillo, 0,225»**.
   - En la página (30/10) es **«no se sostuvo, rojo, 0,199»**, porque se recalcula con los datos nuevos y con un Spearman promediado por cantidad de filas, en vez de la mediana del Pearson.
   - **H3** también cambia: −0,70 (promedio) en el notebook y **−0,90 (mediana)** en la página.

   La página aclara que son números recalculados sobre el 30/10 y cita los de la Entrega 2. Si preguntan: «es la misma hipótesis medida con cinco veces más filas; en los dos casos el efecto propio de la dificultad de drop es chico, y en el modelo aporta 0,02 de R²».
3. **Lo que nos marcaron en la Entrega 3.** En el notebook están documentadas dos correcciones (R² y el error en dólares). Si la devolución tuvo algo más, hay que mostrarlo resuelto en el minuto 0–2.
4. **La decisión de «sólo armas».** Está aplicada en el pipeline y en la URL de Steam («Decisión de alcance 1»), pero no hay una frase que la justifique. Una propuesta: «cuchillos y guantes son otro mercado: no siguen la escalera de rarezas de las armas (salen del slot especial de 0,26 %) y sus precios están en otra escala».
5. **La página tiene que abrir en otra computadora.** El instructivo dice que si la URL no abre el día de la entrega, no hay entrega. También conviene **commitear los cambios pendientes**: hoy hay modificaciones sin commit en `build_app.py`, `app/plantilla.html` y `docs/index.html`. Mientras no se haga push, **GitHub Pages sigue mostrando la versión del 11/11**.
