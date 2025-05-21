"""Taller evaluable presencial"""

#
# Este codigo implementa el algoritmo 'fingerprint' para colisión de textos, el
# cual es utilizado para unificar cadenas de texto que representan la misma
# entidad. Este método permite identificar textos similares a pesar de tener
# pequeñas diferencias en su escritura (mayúsculas, puntuación, orden de palabras, etc.)
# para unificarlos en una sola versión.
#
# El algoritmo funciona en estos pasos principales:
# 1. Normalización: convierte todos los textos a un formato estándar (minúsculas, sin puntuación)
# 2. Tokenización: divide el texto en palabras individuales
# 3. Stemming: reduce las palabras a su raíz léxica
# 4. Ordenamiento y eliminación de duplicados: para que el orden de las palabras no afecte
# 5. Agrupación: textos con la misma "huella digital" (fingerprint) se consideran iguales
#
# Referencia:
# https://openrefine.org/docs/technical-reference/clustering-in-depth
#
# Librerías necesarias:
import nltk  # Librería para procesamiento de lenguaje natural
import pandas as pd  # Librería para manipulación de datos tabulares


def load_data(input_file):
    """Lea el archivo usando pandas y devuelva un DataFrame
    
    Args:
        input_file (str): Ruta al archivo que contiene los datos de entrada
        
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados del archivo
    """
    # Lee el archivo CSV utilizando pandas y lo almacena en un DataFrame
    df = pd.read_csv(input_file)
    return df


def create_normalized_key(df):
    """Cree una nueva columna en el DataFrame que contenga
    el key de la columna 'raw_text'
    
    Esta función implementa el proceso de fingerprinting:
    1. Normaliza el texto (quita espacios, convierte a minúsculas)
    2. Elimina signos de puntuación
    3. Divide en palabras (tokens)
    4. Reduce palabras a su raíz (stemming)
    5. Ordena y elimina duplicados
    6. Une todo en una cadena final normalizada
    
    Args:
        df (pandas.DataFrame): DataFrame con la columna 'raw_text'
        
    Returns:
        pandas.DataFrame: DataFrame con la nueva columna 'key' normalizada
    """
    # Crea una copia del DataFrame para evitar modificar el original
    df = df.copy()

    # Paso 1: Copie la columna 'raw_text' a la columna 'key'
    # Esta será la columna que transformaremos para crear el fingerprint
    df["key"] = df["raw_text"]

    # Paso 2: Remueva los espacios en blanco al principio y al final de la cadena
    # Esto elimina espacios innecesarios que podrían causar diferencias
    df["key"] = df["key"].str.strip()

    # Paso 3: Convierta el texto a minúsculas
    # Para que "Texto" y "texto" se consideren iguales
    df["key"] = df["key"].str.lower()

    # Paso 4: Transforme palabras que pueden (o no) contener guiones por su
    # version sin guion (este paso es redundante por la linea siguiente.
    # Pero es claro anotar la existencia de palabras con y sin '-'.
    df["key"] = df["key"].str.replace("-", "")

    # Paso 5: Remueva puntuación y caracteres de control
    # Esto elimina todos los símbolos de puntuación para que no afecten la comparación
    df["key"] = df["key"].str.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    )

    # Paso 6: Convierta el texto a una lista de tokens (palabras individuales)
    # Divide la cadena en palabras separadas para procesarlas individualmente
    df["key"] = df["key"].str.split()

    # Paso 7: Transforme cada palabra con un stemmer de Porter
    # El stemming reduce palabras a su raíz léxica (ej: "corriendo", "correr", "corrió" -> "corr")
    stemmer = nltk.PorterStemmer()
    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])

    # Paso 8: Ordene la lista de tokens y remueve duplicados
    # Para que el orden de las palabras no afecte la comparación y evitar palabras repetidas
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))

    # Paso 9: Convierta la lista de tokens a una cadena de texto separada por espacios
    # Convierte la lista de palabras procesadas de nuevo a una cadena de texto
    df["key"] = df["key"].str.join(" ")

    return df

def generate_cleaned_text(df):
    """Crea la columna 'cleaned_text' en el DataFrame
    
    Esta función usa la columna 'key' generada previamente para agrupar textos similares.
    Para cada grupo de textos con el mismo 'key' (fingerprint), selecciona el primer 
    texto como representante (cleaned_text) para todo el grupo.
    
    Args:
        df (pandas.DataFrame): DataFrame con las columnas 'raw_text' y 'key'
        
    Returns:
        pandas.DataFrame: DataFrame con la nueva columna 'cleaned_text'
    """
    # Crea una copia del DataFrame para trabajar sin modificar el original
    keys = df.copy()

    # Paso 1: Ordene el dataframe por 'key' y 'raw_text'
    # Agrupa textos con el mismo fingerprint y los ordena alfabéticamente
    keys = keys.sort_values(by=["key", "raw_text"], ascending=[True, True])

    # Paso 2: Seleccione la primera fila de cada grupo de 'key'
    # Para cada grupo de textos con el mismo fingerprint, selecciona el primero
    # como representante de todo el grupo (generalmente el más corto o simple)
    keys = df.drop_duplicates(subset="key", keep="first")

    # Paso 3: Cree un diccionario con 'key' como clave y 'raw_text' como valor
    # Crea un mapeo entre cada fingerprint único y su texto representante
    key_dict = dict(zip(keys["key"], keys["raw_text"]))

    # Paso 4: Cree la columna 'cleaned_text' usando el diccionario
    # Asigna a cada texto original su versión "limpia" según su fingerprint
    # Todos los textos con el mismo fingerprint reciben el mismo texto limpio
    df["cleaned_text"] = df["key"].map(key_dict)

    return df


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo
    
    Selecciona solo las columnas relevantes para el resultado final
    y las guarda en un archivo CSV.
    
    Args:
        df (pandas.DataFrame): DataFrame con las columnas 'raw_text' y 'cleaned_text'
        output_file (str): Ruta donde se guardará el archivo de salida
    """
    # Crea una copia del DataFrame para evitar modificar el original
    df = df.copy()
    
    # Selecciona solo las columnas relevantes para el output final
    # Excluye la columna 'key' que solo se usó internamente para el procesamiento
    df = df[["raw_text", "cleaned_text"]]
    
    # Guarda el DataFrame resultante en un archivo CSV sin incluir el índice
    df.to_csv(output_file, index=False)

def main(input_file, output_file):
    """Ejecuta la limpieza de datos
    
    Función principal que orquesta todo el proceso de limpieza de datos:
    1. Carga los datos
    2. Crea las claves normalizadas (fingerprints)
    3. Genera el texto limpio
    4. Guarda los datos procesados
    
    Args:
        input_file (str): Ruta al archivo de entrada con los datos originales
        output_file (str): Ruta donde se guardará el archivo con los datos procesados
    """
    # Paso 1: Carga los datos del archivo de entrada
    df = load_data(input_file)
    
    # Paso 2: Crea la columna 'key' con el fingerprint de cada texto
    df = create_normalized_key(df)
    
    # Paso 3: Genera la columna 'cleaned_text' con los textos limpios
    df = generate_cleaned_text(df)
    
    # Paso 4: Guarda una copia completa de los datos para referencia
    df.to_csv("files/test.csv", index=False)
    
    # Paso 5: Guarda solo las columnas relevantes en el archivo de salida
    save_data(df, output_file)
    
    # Muestra el DataFrame final en la consola para verificación
    print(df)


if __name__ == "__main__":
    # Este bloque solo se ejecuta cuando el script se ejecuta directamente
    # (no cuando se importa como módulo)
    
    # Llama a la función principal con las rutas de archivos predeterminadas
    main(
        input_file="files/input.txt",  # Archivo con los datos originales
        output_file="files/output.txt",  # Archivo donde se guardarán los resultados
    )
