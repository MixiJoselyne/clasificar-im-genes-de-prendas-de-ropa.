# CLASIFICACIÓN DE IMÁGENES DE PRENDAS DE VESTIR 
Clasificar automáticamente el catálogo de prendas que vende la marca.
# FASHION-MNIST 
El conjunto de datos de imágenes utilizado comúnmente para tareas de clasificación de imágenes. Contiene 70,000 imágenes en escala de grises de 28x28 píxeles que representan prendas de vestir de 10 categorías diferentes. 
# CREACIÓN FASHION-MNIST 
Fashion-MNIST fue creado como una alternativa al conjunto de datos MNIST, que se ha utilizado ampliamente como ejemplo en la comunidad de aprendizaje automático. El conjunto de datos MNIST contiene imágenes de dígitos escritos a mano, y ha sido utilizado durante muchos años como un conjunto de datos estándar para la clasificación de imágenes en el campo de la visión por computadora.
A pesar de que el conjunto de datos MNIST ha sido útil para la investigación y la enseñanza en el campo de la visión por computadora, ha sido criticado por no representar de manera realista el problema de la clasificación de imágenes en el mundo real. Por ejemplo, las imágenes de MNIST son en blanco y negro y tienen una resolución baja, lo que no representa bien la diversidad de las imágenes en el mundo real.
Es por eso por lo que se creó Fashion-MNIST, que es un conjunto de datos más desafiante para las tareas de clasificación de imágenes y representa mejor la diversidad de las imágenes en el mundo real. Con Fashion-MNIST, los investigadores y estudiantes pueden trabajar en problemas de clasificación de imágenes más complejos y realistas, lo que les permite aprender y mejorar sus habilidades en el campo de la visión por computadora.
# APRENDIZAJE AUTOMÁTICO
Como modelo de referencia, MNIST ha sido utilizado ampliamente en la comunidad de aprendizaje automático durante muchos años, y ha sido útil para la investigación y la enseñanza en el campo de la visión por computadora. Sin embargo, existen algunas razones por las cuales es importante considerar la posibilidad de reemplazar MNIST por conjuntos de datos más desafiantes y realistas, Aquí hay algunas razones para reemplazar MNIST:
* Limitaciones en la complejidad de los modelos: Dado que MNIST es un conjunto de datos relativamente simple, es posible que los modelos que se entrenan en él no sean lo suficientemente complejos como para manejar problemas más desafiantes y realistas de clasificación de imágenes.
* Limitaciones en la generalización: Debido a que MNIST es un conjunto de datos relativamente simple y bien estructurado, los modelos que se entrenan en él pueden no generalizar bien a conjuntos de datos más complejos y menos estructurados en el mundo real.
* Necesidad de conjuntos de datos más realistas: Con la creciente cantidad de imágenes y videos disponibles en línea, es importante tener conjuntos de datos más desafiantes y realistas que representen mejor la diversidad de imágenes en el mundo real. De esta manera, los investigadores y estudiantes pueden trabajar en problemas más complejos y realistas de clasificación de imágenes, lo que les permite aprender y mejorar sus habilidades en el campo de la visión por computadora.

Por lo tanto, es importante considerar la posibilidad de reemplazar MNIST con conjuntos de datos más desafiantes y realistas, como Fashion-MNIST, que proporciona una alternativa más compleja y realista para la clasificación de imágenes.
# DESARROLLO 
### OBTENER LOS DATOS
existen varias bibliotecas y herramientas de aprendizaje automático que incluyen el conjunto de datos Fashion-MNIST, lo que hace que sea fácil acceder a él. Aquí hay algunas formas de obtener los datos:
Usando la biblioteca Tensorflow, la biblioteca Tensorflow de Google incluye el conjunto de datos Fashion-MNIST. Puede acceder a él usando la siguiente función: 
```
tf.keras.datasets.fashion_mnist.load_data() 
```
Esta función devuelve dos tuplas, una con los datos de entrenamiento y otra con los datos de prueba.

| Nombre | Descripción | Tamaño |
|:--------------|:-------------:|--------------:|
| train-images-idx3-ubyte.gz | Imágenes del conjunto de entrenamiento  | 26 MBytes |
| train-labels-idx1-ubyte.gz | Etiquetas de conjuntos de entrenamiento  | 29 KBytes |
| t10k-images-idx3-ubyte.gz | Imágenes del conjunto de pruebas  | 4,3 MBytes |
| t10k-labels-idx1-ubyte.gz | Etiquetas de conjuntos de prueba  | 5,1 KBytes |

## ETIQUETAS – PRENDAS DE VESTIR 
Cada ejemplo de entrenamiento y prueba en el conjunto de datos Fashion-MNIST está etiquetado con una de las siguientes etiquetas:
| Etiqueta | Descripción | 
|:--------------|:-------------:|
| 0| Camiseta  | 
| 1 | Pantalón | 
| 2 | Jersey | 
| 3 | Vestido  | 
| 4 | Abrigo  | 
| 5 | Sandalia  | 
| 6 | Camisa  | 
| 7| Sneaker  | 
| 8 | Bolsa  | 
| 9 | Botín  | 

Estas etiquetas se usan para indicar la clase de objeto que aparece en la imagen correspondiente. Por ejemplo, si una imagen está etiquetada como "Camiseta/top" (0), esto significa que la imagen muestra una camiseta o un top.
Es importante tener en cuenta estas etiquetas al trabajar con el conjunto de datos Fashion-MNIST, ya que se utilizan para evaluar la precisión de los modelos de clasificación de imágenes entrenados en este conjunto de datos.
## CARGA DE DATOS EN PYTHON
El código que has compartido es una forma de cargar los datos del conjunto de datos Fashion-MNIST en Python. Aquí está cómo puede cargar los datos usando este enfoque:
```
(train_images, train_labels),(test_images, test_labels)=keras.datasets.fashion_mnist.load_data()
```
La función load_mnist toma dos argumentos: el directorio donde se encuentran los datos del conjunto de datos Fashion-MNIST y el tipo de datos que se desea cargar ("train" para los datos de entrenamiento y "t10k" para los datos de prueba).

La función devuelve dos matrices NumPy (X_train y X_test) que contienen los datos de imagen (cada fila es una imagen) y dos matrices NumPy (y_train e y_test) que contienen las etiquetas correspondientes (cada elemento es una etiqueta de clase).
Ahora puede usar las matrices X_train, y_train, X_test y y_test para entrenar y evaluar modelos de aprendizaje automático en el conjunto de datos Fashion-MNIST.
## PROCESADOR DE DATOS EN PYTHON
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
El código que se muestra está relacionado con el preprocesamiento de imágenes en aprendizaje automático. Específicamente, se divide cada valor de píxel en las imágenes de entrenamiento y prueba por 255, que es el valor máximo posible de un píxel en una imagen en escala de grises o en un canal en una imagen a color. Esto tiene el efecto de escalar los valores de píxeles en un rango de 0 a 1, lo que puede ser beneficioso para algunos algoritmos de aprendizaje automático.

El preprocesamiento de datos es una parte importante en el aprendizaje automático, ya que puede mejorar la calidad y eficacia de los modelos que se crean. En este caso, el preprocesamiento de imágenes se realiza antes de entrenar un modelo de aprendizaje automático para que el modelo pueda trabajar con valores de píxeles escalados y normalizados, lo que puede mejorar la precisión y velocidad de entrenamiento.
## DEFINICIÓN DE LA ARQUITECTURA DEL MODELO 
```
model = keras.Sequential(
    [
       
    keras.layers.Conv2D(32, (3, 3), padding='same', activation ='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),                      
    ]
)
model.summary()

```
El código que se muestra define la arquitectura de un modelo de red neuronal convolucional, utilizando la biblioteca Keras de Python. La arquitectura del modelo consta de varias capas que se apilan una encima de la otra:
* Capa de convolución: La primera capa es una capa convolucional que utiliza un filtro de tamaño 3x3 y tiene 32 canales. La capa usa una función de activación "relu" y recibe una entrada de forma (28, 28, 1), que corresponde a imágenes en escala de grises de 28x28 píxeles.
* Capa de MaxPooling: La segunda capa es una capa de MaxPooling que reduce la dimensión espacial de la salida de la capa anterior.
* Capa de convolución: La tercera capa es otra capa convolucional que utiliza un filtro de tamaño 3x3 y tiene 64 canales. La capa también usa una función de activación "relu" y utiliza el mismo tipo de padding que la capa anterior.
* Capa de MaxPooling: La cuarta capa es otra capa de MaxPooling que reduce aún más la dimensión espacial de la salida de la capa anterior.
* Capa de convolución: La quinta capa es una capa convolucional adicional que utiliza un filtro de tamaño 3x3 y tiene 128 canales. La capa usa una función de activación "relu" y el mismo tipo de padding que las capas anteriores.
* Capa de aplanamiento: La sexta capa es una capa de aplanamiento que toma la salida de la capa anterior y la convierte en un vector 1D para que pueda ser alimentada a una capa completamente conectada.
* Capa completamente conectada: La séptima capa es una capa densa con 128 neuronas y utiliza una función de activación "relu".
* Capa de salida: La última capa es otra capa densa con 10 neuronas, que corresponde al número de clases diferentes en el conjunto de datos de MNIST. Utiliza una función de activación "softmax" para producir probabilidades de pertenencia a cada clase.
La función "summary()" muestra una descripción resumida de la arquitectura del modelo, incluyendo el número de parámetros y el tamaño de cada capa.
## ENTRENAMIENTO DEL MODELO
```
model.fit(train_images.reshape((-1,28,28,1)), train_labels, epochs=10)
```
El código que se muestra entrena el modelo utilizando los datos de entrenamiento y las etiquetas correspondientes. La función "fit" ajusta los pesos del modelo utilizando un algoritmo de optimización y los datos de entrenamiento. En este caso, el modelo se entrena durante 10 épocas. La función "reshape" para ajustar las dimensiones de las imágenes de entrenamiento a (28, 28, 1) para que coincidan con las dimensiones de entrada especificadas en la primera capa de la red.

El entrenamiento del modelo puede llevar varios minutos, dependiendo del tamaño del conjunto de datos y la complejidad del modelo. Después del entrenamiento, el modelo debería haber mejorado su capacidad para clasificar correctamente las imágenes de entrada.
## EVALUAR EL MODELO DEL CONJUNTO DE PRUEBA
```
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\n Test_acc:', test_acc)
print('\n Test_loss:', test_loss)
```
El código que se muestra evalúa el modelo utilizando el conjunto de datos de prueba después de entrenarlo. La función "evaluate" toma como entrada el conjunto de imágenes de prueba y las etiquetas correspondientes y devuelve dos valores: la pérdida (loss) y la precisión (accuracy) del modelo en el conjunto de prueba.

El código imprime el valor de precisión y pérdida en el conjunto de prueba utilizando las variables "test_acc" y "test_loss", respectivamente.
Es importante evaluar el modelo en un conjunto de datos separado para verificar si ha aprendido a generalizar bien en nuevos datos que no ha visto durante el entrenamiento. En general, un modelo que tenga una alta precisión en el conjunto de datos de prueba indica que está generalizando bien y que es un modelo efectivo.
## CLASIFICACIÓN DE IMÁGENES DE PRENDA DE VESTIR
```
test_loss, test_acc = model.evaluate(test_images, test_labels)

fig, axs = plt.subplots(4, 4, figsize=(10, 10))
axs = axs.flatten()
for i in range(16):
    imagen = test_images[i].reshape(28, 28)
    axs[i].imshow(imagen, cmap='gray')
    axs[i].set_title(f"Tipo de prenda: {prendas.get(np.argmax(predictions[i]))}")
plt.tight_layout()
plt.show()
```
El código que se muestra es para visualizar algunas de las imágenes de prueba junto con las etiquetas predichas por el modelo después de ser entrenado.

El código utiliza la función "subplots" para crear una matriz de gráficos de 4x4 y la función "flatten" para aplanar la matriz de ejes en un arreglo unidimensional. Luego, utiliza un bucle para recorrer los primeros 16 elementos del conjunto de prueba y muestra cada imagen junto con la etiqueta predicha por el modelo utilizando la función "imshow" y el método "set_title" del objeto de gráfico correspondiente. La función "get" del diccionario se utiliza para obtener la etiqueta de prenda correspondiente a la etiqueta de clase predicha por el modelo.

![prendas](https://user-images.githubusercontent.com/107889451/222944477-5a3dbd1b-e360-4945-b557-b2c3147243bf.JPG)

