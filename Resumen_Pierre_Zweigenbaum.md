# Resumen: Information extraction from biomedical text

### Integrantes:
* Centeno Cerna, Alexander Jesús
* Del Carpio Villacrés, Luis André
* Goicochea Contreras, Brayan Emir
* Saavedra Nieto, Rissel Aaron

---

### 1. La Era Simbólica: Las "Reglas Artesanales" ✍️

En los inicios, literalmente escribíamos cientos de reglas a mano para capturar información.

* **Ejemplo de Regla para Detección de Fiebre:**
    Si queríamos detectar fiebre en un texto clínico, una regla simple (simplificada aquí) se vería así:

    > `SI (palabra_actual == "temperatura" Y palabra_siguiente > "38") O (palabra_actual == "fiebre")`
    > `ENTONCES marcar como [SÍNTOMA: FIEBRE]`

    *El problema práctico:* Esta regla fallaba si el médico escribía "El paciente está **febril**" (necesitábamos otra regla) o "No tiene **fiebre**" (falso positivo grave).

---

### 2. El Poder del UMLS: La "Piedra Rosetta" Médica 🪨

El UMLS fue revolucionario porque nos permitió normalizar. Antes de él, cruzar datos entre hospitales era casi imposible si usaban vocabularios distintos.

* **Ejemplo de Normalización:**
    Diferentes médicos pueden escribir lo mismo de formas muy distintas:
    * "Infarto agudo de miocardio"
    * "Ataque cardíaco"
    * "IAM" (acrónimo)

    El UMLS nos permite asignar a *todas* estas variantes el mismo Identificador Único de Concepto (CUI): **C0155626**. Esto es vital para que una computadora entienda que tres pacientes diferentes tienen exactamente la misma patología.

---

### 3. El Desafío de la Negación y los "Antecedentes" 🚫

En medicina, lo que *no* tiene el paciente es tan importante como lo que sí tiene.

* **El problema de la "Bolsa de Palabras" (Bag-of-Words):**
    Un sistema estadístico primitivo vería la frase:

    > "Madre falleció de **cáncer de mama**."

    Y podría clasificar erróneamente al paciente actual como enfermo de cáncer de mama.

* **La solución temprana (Algoritmo ConText):**
    Desarrollamos algoritmos específicos (como *NegEx* o *ConText*) que buscaban "ventanas" de contexto. Si aparecía "Madre de..." o "Antecedentes de...", el sistema sabía que el concepto siguiente **no** pertenecía al estado actual del paciente, sino a su historial familiar.

---

### 4. Ambigüedad: El Dolor de Cabeza Diario 🤯

El lenguaje médico está lleno de acrónimos que cambian según el departamento del hospital.

* **Ejemplo del acrónimo "IR":**
    * En Nefrología, "IR" significa casi siempre **Insuficiencia Renal**.
    * En Neumología, "IR" significa **Insuficiencia Respiratoria**.

    Un sistema antiguo basado solo en diccionarios fallaría estrepitosamente aquí. Necesitábamos empezar a mirar el contexto del documento (¿quién lo firma? ¿qué otras palabras aparecen?) para desambiguar.

---

### 5. El Pragmatismo Moderno: ¿Por qué BioBERT cuando hay pocos datos? 🚀

Durante décadas, nuestro mayor obstáculo fue el "arranque en frío": si un hospital pequeño quería un sistema para detectar una enfermedad rara en sus notas clínicas, no tenía los millones de documentos necesarios para entrenar un modelo estadístico desde cero.

Aquí es donde entra la revolución del *Transfer Learning* (aprendizaje por transferencia) y modelos como BioBERT.

#### El Problema de los Modelos Generalistas (como el BERT básico)

Imaginen un modelo de lenguaje genérico (entrenado con Wikipedia y libros) como un estudiante de secundaria muy inteligente. Sabe leer y escribir perfectamente, pero si le das una nota clínica compleja con términos como “acetilcolinesterasa” o “carcinoma hepatocelular”, tropezará. Puede que divida mal las palabras o no entienda el contexto de gravedad implícito.

#### La Solución: BioBERT, el "Estudiante de Medicina"

BioBERT no empieza de cero. Toma ese "estudiante inteligente" (BERT) y lo obliga a leer millones de resúmenes de PubMed y artículos completos de PubMed Central.

* **Resultado:** Cuando nosotros, en un laboratorio o un hospital con recursos limitados, usamos BioBERT, ya estamos trabajando con un "residente de medicina". Ya sabe que la *metformina* es un fármaco y suele aparecer cerca de la *diabetes*.
