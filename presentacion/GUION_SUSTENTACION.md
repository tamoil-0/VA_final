# Guion de sustentación — Dashboard de riesgo agroclimático por heladas en Puno

**Duración objetivo:** 10 minutos de exposición + preguntas

**Orden:** Juan Diego Canaza Paucara → Angie Tatiana Luque Pacheco → Jhon Marco Aracayo Mamani

**Reparto:** Juan Diego, diapositivas 1–6; Angie, diapositivas 7–10; Jhon, diapositivas 11–14.

> Este texto está pensado para hablar con naturalidad. No se deben leer las tablas ni repetir todos los elementos visibles: se señala el dato principal, se explica qué significa y se enlaza con la siguiente idea.

## 1. Juan Diego Canaza Paucara — contexto, preguntas y datos (3 min 40 s)

### Diapositiva 1. Portada — 20 s

**Texto para decir**

“Buenos días. Somos Juan Diego Canaza, Angie Tatiana Luque y Jhon Marco Aracayo. Presentaremos nuestro proyecto sobre el riesgo agroclimático por heladas en la región Puno. Nuestro producto principal es un dashboard interactivo que convierte diez años de datos climáticos diarios en información útil para reconocer cuándo, dónde y con qué intensidad se concentra el riesgo.”

**Paso:** avanzar de inmediato; no leer los datos institucionales de la portada.

### Diapositiva 2. Recorrido de la exposición — 25 s

**Texto para decir**

“La exposición seguirá el mismo razonamiento con el que construimos el proyecto. Primero plantearemos el problema y las preguntas; luego explicaremos los datos y su preparación; después mostraremos cómo funciona el dashboard; y finalmente presentaremos los resultados, las decisiones que permite orientar y sus limitaciones. La idea central que queremos demostrar es que el riesgo no es uniforme en Puno: cambia claramente según el momento del año y la provincia.”

### Diapositiva 3. El problema — 45 s

**Texto para decir**

“Las heladas afectan cultivos altoandinos como papa, quinua y cañihua, además de la seguridad alimentaria y la actividad ganadera. El problema no es que falten datos climáticos, sino que existen en grandes volúmenes y en formatos que dificultan interpretarlos para una decisión concreta. Por eso formulamos esta pregunta: ¿cómo se distribuyen la frecuencia y la intensidad de las heladas en las trece provincias de Puno y qué factores climáticos están asociados con ellas? Nuestra respuesta no es solamente un informe estático: es una herramienta que permite explorar el periodo 2015 a 2024 y comparar territorios y umbrales.”

### Diapositiva 4. Las seis preguntas de análisis — 35 s

**Texto para decir**

“Organizamos todo el análisis alrededor de seis preguntas. Empezamos por la situación general; luego estudiamos la distribución temporal y provincial, la tendencia entre años y el ranking territorial. Después analizamos la relación con variables como altitud, humedad y nubosidad, y terminamos con técnicas multivariantes para encontrar factores latentes, grupos y días atípicos. Este esquema fue también nuestro criterio visual: cada gráfico debía responder una pregunta; si no aportaba una respuesta, no se incorporaba.”

### Diapositiva 5. El conjunto de datos y su procedencia — 50 s

**Texto para decir**

“Trabajamos con dos fuentes independientes. La fuente principal es ERA5-Land y ERA5, de Copernicus y ECMWF, con una resolución aproximada de nueve kilómetros. MERRA-2, de NASA POWER, se utilizó como validación externa y tiene una malla mucho más gruesa, cercana a cincuenta y cinco kilómetros. Esta diferencia fue decisiva: con MERRA-2, Chucuito, El Collao y Yunguyo compartían una misma celda y producían series idénticas; incluso la altitud asignada a Sandia no representaba su capital. Por eso habría sido incorrecto usar esa fuente para un ranking provincial. El conjunto final contiene 47 mil 489 registros: son 3 mil 653 días por trece provincias, con 52 variables y sin datos personales.”

### Diapositiva 6. Limpieza, transformación e integración — 45 s

**Texto para decir**

“Antes de visualizar, aplicamos un pipeline auditable de trece etapas. Normalizamos nombres y fechas, unificamos unidades, validamos rangos físicos y coherencia térmica, controlamos duplicados, completamos el calendario por provincia, integramos la fuente secundaria y generamos 23 variables derivadas. Un control especialmente importante fue el de radiación solar: los datos estaban en megajulios por metro cuadrado y día. Si hubiéramos usado un límite pensado para kilovatios-hora, habríamos anulado 46 mil 564 observaciones válidas y luego las habríamos rellenado artificialmente. Este caso muestra por qué la calidad del análisis depende primero de comprender las unidades.”

**Transición a Angie**

“Con los datos ya consistentes y trazables, Angie explicará cómo los convertimos en una experiencia visual e interactiva.”

## 2. Angie Tatiana Luque Pacheco — dashboard y resultados descriptivos (3 min 10 s)

### Diapositiva 7. Arquitectura del software — 40 s

**Texto para decir**

“El dashboard fue construido por capas. La configuración concentra rutas, metadatos y criterios comunes; las utilidades contienen el procesamiento, la estadística y los modelos; los componentes reutilizables se encargan de indicadores, gráficos y descargas; y las páginas solamente coordinan esos elementos. Esta separación evita repetir reglas y permite probar la lógica sin depender de la interfaz. En el diseño usamos una identidad pastel naranja y azul, pero con tonos de datos que conservan contraste y diferenciación, incluso ante protanopía y deuteranopía. También limitamos las comparaciones visuales cuando demasiadas series podían volver ambigua la lectura.”

### Diapositiva 8. Demostración en vivo — 50 s

**Acción y texto para decir**

“En la vista general encontramos los indicadores y un resumen que cambia con la selección. Por ejemplo, al alternar el umbral de cero grados, que representa la helada meteorológica, al de tres grados, que representa el riesgo agronómico, cambia inmediatamente la escala del fenómeno. Los ocho filtros son compartidos: si elegimos una provincia o un periodo, la selección se mantiene al pasar al análisis descriptivo, multidimensional y geográfico. Los gráficos y sus interpretaciones se recalculan con ese mismo subconjunto. Finalmente, el usuario puede descargar los datos filtrados en CSV o en un libro Excel con varias hojas.”

**Plan de contingencia:** si el dashboard demora o no abre, no esperar. Señalar la captura correspondiente y decir: “La evidencia completa de las seis vistas se incluye en el repositorio”; continuar con la diapositiva 9.

### Diapositiva 9. Resultados: situación general — 45 s

**Texto para decir**

“En todo el periodo analizamos 47 mil 489 observaciones provincia-día. Encontramos 7 mil 620 heladas meteorológicas, equivalentes al 16,05 por ciento. Al aplicar el umbral agronómico de tres grados aparecen 18 mil 877 observaciones de riesgo, es decir, 39,75 por ciento. La exposición potencial para los cultivos es, por tanto, aproximadamente 2,48 veces la que mostraría el umbral físico estricto. Entre las heladas meteorológicas predominan las ligeras, con 5 mil 180 casos; hubo 2 mil 90 moderadas, 320 severas y 30 extremas. Esto indica que planificar únicamente con cero grados subestimaría de forma importante el riesgo agrícola.”

### Diapositiva 10. Resultados: desigualdad territorial — 55 s

**Texto para decir**

“La comparación provincial confirma que Puno no tiene un riesgo homogéneo. Carabaya encabeza el ranking con mil 126 días de helada; le siguen San Antonio de Putina con 976, El Collao con 962 y San Román con 930. En cambio, en los puntos de malla analizados, Sandia y Yunguyo no registraron heladas meteorológicas. El contraste más útil es Yunguyo frente a San Román: ambas localidades están prácticamente a la misma altitud, alrededor de 3 mil 825 metros, pero presentan cero frente a 930 días. Esto muestra que la altitud por sí sola no basta y que el patrón es consistente con el efecto termorregulador del lago Titicaca en las zonas circunlacustres. En consecuencia, una política basada únicamente en pisos altitudinales produciría una focalización deficiente.”

**Transición a Jhon**

“Hasta aquí observamos cuánto y dónde ocurre el fenómeno. Para cerrar, Jhon presentará la robustez estadística, las conclusiones y el alcance real de estos resultados.”

## 3. Jhon Marco Aracayo Mamani — análisis avanzado y cierre (3 min 10 s)

### Diapositiva 11. Tendencia y técnicas avanzadas — 1 min 10 s

**Texto para decir**

“Para la evolución anual usamos dos métodos. La regresión OLS estimó una disminución de 1,37 días de helada por provincia y año, pero su valor p fue 0,132. Mann-Kendall también dio una dirección descendente, con una pendiente de Sen de menos un día por año, pero su valor p fue 0,107. Como ambos superan 0,05, no afirmamos una tendencia estadísticamente significativa ni atribuimos cambio climático con solo diez años.

También contrastamos las fuentes: ERA5-Land y MERRA-2 alcanzaron una correlación de 0,611, un sesgo medio de más 1,29 grados y coincidieron en el 83,06 por ciento de la clasificación de helada. En el análisis multivariante, los dos primeros componentes del PCA explicaron 68,66 por ciento de la variación y cinco componentes llegaron al 90 por ciento. K-Means encontró tres grupos, pero su silueta de 0,306 indica una estructura débil, por lo que no presentamos esos grupos como fronteras rígidas. Isolation Forest marcó 475 observaciones para revisión. Finalmente, el Random Forest obtuvo un AUC de 0,990 con partición cronológica y excluyendo la temperatura mínima que define el evento; se usa para interpretar factores asociados, no como pronóstico operativo.”

### Diapositiva 12. Conclusiones — 50 s

**Texto para decir**

“Concluimos, primero, que la heterogeneidad territorial es la característica dominante: no corresponde tratar a las trece provincias como una sola zona de riesgo. Segundo, el umbral agronómico cambia la decisión, porque eleva la proporción observada de 16,05 a 39,75 por ciento. Tercero, la altitud influye, pero no explica por sí sola contrastes como Yunguyo y San Román; deben considerarse humedad, nubosidad y cercanía al lago. Cuarto, la mayor exposición se concentra en la estación seca: julio alcanza una tasa mensual de 49,93 por ciento. Finalmente, la concordancia de 83,06 por ciento entre dos reanálisis independientes respalda la existencia de la señal general, aun cuando los valores locales deben interpretarse con cautela.”

### Diapositiva 13. Recomendaciones y limitaciones — 55 s

**Texto para decir**

“Estos resultados permiten recomendar una focalización provincial del seguro agrario y de las medidas de protección, calendarios de siembra diferenciados y monitoreo reforzado durante la ventana seca. También sugieren priorizar mediciones de superficie donde las fuentes discrepan más. Sin embargo, declaramos límites claros: ERA5-Land tiene una malla aproximada de nueve kilómetros y suaviza extremos locales; cada provincia está representada por un punto cercano a su capital; el periodo de diez años no constituye una normal climática de treinta años; no contamos con rendimiento agrícola para estimar daño económico; y los pesos del índice de riesgo responden a criterio experto, no a una calibración con pérdidas observadas. Por ello, el dashboard apoya decisiones, pero no reemplaza las alertas oficiales ni las estaciones de SENAMHI.”

### Diapositiva 14. Cierre — 15 s

**Texto para decir**

“En síntesis, entregamos datos trazables, un análisis reproducible y una herramienta interactiva para convertir la heterogeneidad climática de Puno en decisiones mejor focalizadas. Muchas gracias por su atención. Quedamos atentos a sus preguntas.”

## Control de tiempo

| Momento | Tiempo acumulado esperado | Señal de control |
|---|---:|---|
| Juan termina diapositiva 4 | 2:05 | Si pasa de 2:15, resumir la diapositiva 5 sin mencionar el ejemplo de Sandia. |
| Juan entrega a Angie | 3:40 | La transición debe durar una sola frase. |
| Angie termina la demostración | 5:10 | La demo no debe superar 50 segundos; hacer sólo un cambio de umbral y un filtro. |
| Angie entrega a Jhon | 6:50 | Si hay retraso, omitir la enumeración completa del ranking y conservar Carabaya, Yunguyo y San Román. |
| Jhon termina diapositiva 11 | 8:00 | Mencionar como mínimo los valores p de tendencia, la concordancia y la cautela del clustering. |
| Inicio de diapositiva 14 | 9:45 | Cerrar sin volver a resumir cifras. |
| Fin | **10:00** | Pausa, mirada al jurado y apertura de preguntas. |

### Recortes seguros si faltan 30–45 segundos

1. En la diapositiva 4, decir únicamente que las seis preguntas cubren situación, tiempo, territorio y factores asociados.
2. En la diapositiva 7, omitir la explicación de accesibilidad cromática.
3. En la diapositiva 9, omitir el desglose por severidad y conservar 16,05 %, 39,75 % y el factor 2,48.
4. En la diapositiva 11, mencionar PCA, K-Means, anomalías y Random Forest en una sola frase, sin repetir todas sus métricas.

## Posibles preguntas del jurado y respuestas breves

### 1. ¿Por qué eligieron ERA5-Land como fuente principal?

“Porque su resolución aproximada de nueve kilómetros permite diferenciar mejor los puntos provinciales. MERRA-2 tiene una malla cercana a 55 kilómetros y asignaba series idénticas a varias localidades vecinas. Por eso la conservamos como validación independiente, no como base del ranking.”

### 2. ¿Por qué usan dos umbrales, 0 °C y 3 °C?

“Cero grados identifica la helada meteorológica estricta. Tres grados representa un umbral agronómico preventivo, porque un cultivo puede sufrir daño aunque la medición no baje de cero. Separarlos evita confundir el fenómeno físico con el riesgo para la agricultura.”

### 3. ¿Decir que Yunguyo tuvo cero heladas significa que nunca heló en toda la provincia?

“No. Significa que el punto de malla representativo analizado no registró temperaturas mínimas iguales o menores a cero durante el periodo. La malla no resuelve todos los microclimas ni los fondos de valle, por eso declaramos esa limitación.”

### 4. ¿La reducción observada demuestra que las heladas están desapareciendo?

“No. OLS dio una pendiente de menos 1,37 días por año con p igual a 0,132, y Mann-Kendall p igual a 0,107. La dirección estimada es descendente, pero no es estadísticamente significativa. Además, diez años no permiten atribuir cambio climático.”

### 5. ¿Cómo justifican el efecto del lago Titicaca?

“Lo presentamos como una interpretación consistente con los datos, no como una prueba causal. Yunguyo y San Román están casi a la misma altitud y tienen resultados muy distintos; esa comparación muestra que la altitud no basta y es compatible con la inercia térmica del lago. Para estimar causalidad harían falta estaciones y un diseño específico.”

### 6. ¿Qué significa la concordancia de 83,06 % entre fuentes?

“Que ambas fuentes clasifican del mismo modo —helada o no helada— en el 83,06 por ciento de los pares provincia-fecha. No significa que sus temperaturas sean idénticas: su correlación fue 0,611 y el sesgo medio, más 1,29 grados, en parte por la diferencia de resolución.”

### 7. ¿Un AUC de 0,990 en Random Forest no indica fuga de información?

“Controlamos ese riesgo con una partición cronológica, no aleatoria, y excluimos la temperatura mínima porque es la variable que define la etiqueta. Aun así, lo interpretamos como un modelo de asociación sobre este conjunto, no como un sistema operativo de pronóstico.”

### 8. ¿Los tres grupos de K-Means representan zonas climáticas definitivas?

“No. La silueta fue 0,306, que corresponde a una estructura débil. Los grupos sirven para explorar perfiles semejantes, pero no para dibujar fronteras climáticas rígidas ni sustituir conocimiento territorial.”

### 9. ¿Por qué analizaron las capitales y no toda el área de cada provincia?

“Necesitábamos puntos comparables, con coordenadas y altitudes verificables, para cubrir las trece provincias en una serie diaria completa. Es una aproximación reproducible, pero no representa toda la variabilidad interna; por eso proponemos ampliar el trabajo con más estaciones o puntos de malla.”

### 10. ¿El índice de riesgo predice pérdidas agrícolas?

“No. Resume exposición climática en una escala común y sus pesos fueron fijados con criterio experto. Como no contamos con datos de rendimiento o pérdidas, no afirmamos una relación económica calibrada.”

### 11. ¿Qué hicieron con los valores atípicos?

“No los eliminamos automáticamente. Isolation Forest marcó 475 observaciones, aproximadamente el uno por ciento, para revisión. La anomalía señala una combinación poco frecuente de variables, no necesariamente un error.”

### 12. ¿Cuál fue el error de limpieza más peligroso que evitaron?

“La interpretación de la radiación solar. Al verificar que estaba en megajulios por metro cuadrado y día evitamos aplicar un rango de otra unidad que habría invalidado 46 mil 564 observaciones correctas.”

## Acuerdos de presentación

- Juan deja abierto el dashboard antes de comenzar; Angie realiza la demostración y Jhon controla el cronómetro.
- Sólo Angie manipula el equipo durante la demo; los demás mantienen la mirada en el jurado.
- Pronunciar las cifras redondeadas como aparecen en este guion; no improvisar decimales adicionales.
- Al cambiar de persona, quien termina nombra a quien continúa y se aparta sin conversación lateral.
- En preguntas, responde primero quien expuso el tema; Jhon complementa únicamente si hace falta y realiza el cierre final.
