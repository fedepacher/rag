<p align=center><img src=resources/assets/front.png height=500 weight=1247><p>

# Aplicación de un modelo de lenguaje (LLM) a un uso educativo implementado en servidores de pequeño porte

## Descripción del trabajo

El objetivo de este proyecto es adaptar un modelo de lenguaje de forma tal que pueda ser
utilizado para responder consultas en forma asincrónica restringidas al temario de una
asignatura y que pueda correr en un servidor de pequeño porte, adecuado a los recursos
existentes hoy en dia en las instituciones académicas de nuestro país.
De esta manera se podría dotar a dicha asignatura con un recurso educativo más para
ponerlo al servicio de los estudiantes.
Dada la restricción respecto del hardware a utilizar, esto inevitablemente redundará en un
tiempo de respuesta elevado, por lo que las consultas serán via una API Rest y la respuesta sera enviada por email.



## Esquema

Para la implementacion del sistema se propone un esquema de microservicios el cual contara con los siguienetes elementos:

- **API Rest**: Servicio de entrada donde el usuario podra realizar sus preguntas. Este servicio consta de 5 endpoints:
  - Estado de servicio: este endpoint solo sirve para informar que el servicio esta activo.
  - Registro de usuario: permitira que un usuario nuevo se registre para el uso de la aplicacion.
  - Login de usuario: permitira que un usuario previamente registrado pueda interactuar con el sistema de consultas RAG.
  - Creacion de prompt: permitira que un usuario previamente logueado pueda crear una pregunta para que el sistema RAG responda.
  - Recepcion de respuesta: una vez que el sistema RAG de una respuesta esta sera reflejada en el endpoint correspondiente.

- **Base de datos relacional (MySQL)**: El sistema almacenara en una base de datos relacional a cada usuario registrado 
permitiendo que dicho usuario una vez ingresado al sistema pueda realizar una consulta.

- **Base de datos no relacional (MongoDB)**: Cada pregunta realizada y respuesta seran almacenadas en una base de datos 
no relacion de manera tal que dicha base de datos funcione como cola de mensajes (FIFO), permitiendo que el servicio RAG 
colecte de la base de datos preguntas sin respuesta y las vaya resopndiendo a medidas que se libera independizando asi 
los sistemas.

- **RAG (retrieval-augmented generation)**: Servicio que buscara dentro de la base de datos (MongoDB) toda pregunta 
(campo input) cuyo respuesta (output) este marcada como NULL ordenada por forma incremental mediante el campo date. 
De esta manera se respondera las preguntas mas antiguas primero.

- **Ollama server**: Servicio que sera utilizado para poner en en funcionamiento un sistema de LLM el cual es consultado 
- por el servicio RAG.



<p align=center><img src=resources/assets/diagrama.png height=500 weight=1247><p>


## RAG - Retrieval-Augmented Generation

RAG es una técnica para aumentar el conocimiento de los LLM con datos adicionales.

Los LLM pueden razonar sobre temas muy variados, pero su conocimiento se limita a los datos públicos hasta un punto 
específico en el tiempo en el que fueron entrenados. Si desea crear aplicaciones de IA que puedan razonar sobre datos 
privados o datos introducidos después de la fecha límite de un modelo, debe aumentar el conocimiento del modelo con la 
información específica que necesita. El proceso de traer la información adecuada e insertarla en el mensaje del modelo 
se conoce como Generación aumentada de recuperación (RAG).


### Arquitectura RAG

Una aplicación RAG típica tiene dos componentes principales:

Indexación: un canal para ingerir datos de una fuente e indexarlos. Esto suele ocurrir sin conexión.

Recuperación y generación: la cadena RAG real, que toma la consulta del usuario en tiempo de ejecución y recupera los 
datos relevantes del índice, luego los pasa al modelo.

La secuencia completa más común desde los datos sin procesar hasta la respuesta se ve así:

**Indexación**
1. **Carga**: primero debemos cargar nuestros datos. Esto se hace con DocumentLoaders.
2. **División**: los divisores de texto dividen los documentos grandes en fragmentos más pequeños. Esto es útil tanto para 
indexar datos como para pasarlos a un modelo, ya que los fragmentos grandes son más difíciles de buscar y no caben en 
la ventana de contexto finita de un modelo.
3. **Almacenamiento**: necesitamos un lugar para almacenar e indexar nuestras divisiones, de modo que luego se puedan buscar. 
Esto se hace a menudo utilizando un modelo VectorStore e incrustaciones.

<p align=center><img src=resources/assets/data_index.png height=500 weight=900><p>

**Recuperación y generación**
4. **Recuperar**: dada una entrada del usuario, las divisiones relevantes se recuperan del almacenamiento mediante un 
recuperador.
5. **Generar**: un ChatModel/LLM produce una respuesta mediante un mensaje que incluye la pregunta y los datos recuperados

<p align=center><img src=resources/assets/retrieve.png height=500 weight=900><p>


## Base de datos relacional MySQL

Para la creacion de usuario se implemento una tabla columnar `users` que posee las siguiente estructura:
- **id**: Numero de identificacion unico que se genera automaticamente de manera incremental.
- **email**: Email del usuario el cual servira para posteriormente loguearse y ademas sera usado como contacto para 
enviar la respuesta del sistema.
- **username**: Nombre de usuario el cual servira para posteriormente loguearse. 
- **password**: Password del usuario cifrado.


## Base de datos no relacional MongoDB

Cada pregunta (prompt) generada por un usuario sera almacenada en una base de datos no relacional. Esta base de datos 
tiene dos grandes funcionalidades dentro del sistema. Ser un buffer para las entradas, es decir, encolar las preguntas 
de los usuarios para que el sistema pueda responder por orden de ingreso a la base de datos. Por otro lado, se la 
utiliza para recavar informacion para un posterior proceso de entrenamiento de un modelo LLM.
Esta base de datos posee la siguiente estructura:
- **_id**: Numero de identificacion unico que se genera automaticamente de manera incremental.
- **input**: Pregunta que genera el usuario para ser respondida.
- **date**: Hora y fecha de la pregunta realizada.
- **output**: Respuesta del sistema RAG.
- **email**: Email de contacto donde sera enviada la respuesta.
- **status**: Bandera que indica si el mail con la respuesta ha sido enviado al usuario. Los posibles estados son `sent`
que indica que se ha enviado la respuesta o `null` que indica que la respuesta aun no ha sido enviada.


## Flujo de trabajo

El usuario que desee utilizar el sistema debe crearse un usuario (endpoint `user`) y posteriormente loguearse al 
sistema. De esta manera ya se posee informacion necesaria para el proceso de las acciones siguientes. El login del 
usuario se puede realizar a traves del endpoint `login` o mediante el boton `Authorize` ubicado en la parte superior
derecha.
Las imagenes a continuacion muestran los endpoints disponibles en la API para las acciones descriptas anteriormente:

<p align=center><img src=resources/assets/swagger.png height=500 weight=900><p>

Se utiliza Swagger UI para la visualizacion de los endpoints ya que ofrece una visual mas amigable al usuario.
El proceso de login genera un Token que posee una duracion de 24hs, pasado este tiempo el usuario debera loguearse 
nuevamente. El Token es utilizado con el fin de obtener la informacion del usuario y asi prevenir usos indebidos o 
maliciosos en el sistema.

<p align=center><img src=resources/assets/login.png height=500 weight=900><p>

Ademas se han agregado endpoints para recuperacion de contraseña en caso de olvido y para el cambio de la misma. En 
el caso de la recuperacion el sistema generara una nueva contraseña, la cual sera almacenada temporalmente en la base 
de datos reemplazando la anterior, y sera enviada una notificacion al usuario con su nueva contraseña.
Una vez ingresado al sistema el usuario ya esta en condiciones de realizar una consulta. Para dicha tarea se dispone
del endpoint `Input Prompt`.
Toda pregnta realizada por el usuario sera almacenada en la base de datos MongoDB y el sistema respondera que dicha 
consulta ya fue almacenada.

<p align=center><img src=resources/assets/prompt.png height=500 weight=900><p>

Por otro lado el sistema RAG estara consultando el endpoint `receive-prompt` con el fin de obtener cualquier nueva 
pregunta ingresada al sistema. Esto proceso sera realizado mediante el chequeo del estado del campo `output` de
la base de datos Mongo, si dicho campo posee el valor `null` la pregunta sera procesada. Finalizado este proceso
se almacenara la respuesta en la base de datos Mongo.

<p align=center><img src=resources/assets/login.png height=500 weight=900><p>

El sistema de envio de respuesta funciona de manera similar al sistema RAG, estara chequeando la base de datos MongoDB
en busca de los campos `output` distinto de `null` y `status` igual a `null`. Cunado encunetra esta condicion procede 
al envio de la respuesta mediante email y posteriormente actualiza el `status` a `sent`.

<p align=center><img src=resources/assets/email.png height=500 weight=900><p>

## Roadmap

Este proyecto fue planteado con una duracion de 3 meses. El roadmap propuesto se puede encontrar en el siguiente [link](https://github.com/users/fedepacher/projects/7/views/4)

<p align=center><img src=resources/assets/roadmap.png height=500 weight=900><p>

Las tareas propuestas fueron descriptas mediante issues y se pueden encontrar en el siguiente [link](https://github.com/users/fedepacher/projects/7/views/3)
Cada issue consto de una descripcion detallada del problema y diferentes items a ser resultos. Ademas se estimo un 
tiempo de desarrollo una prioridad y una fecha estimada de inicio y finalizacion que se intento cumplir.

<p align=center><img src=resources/assets/issue.png height=500 weight=900><p>

# Puesta en funcionamiento

Para poner el sistema de en funcionamiento se utilizo contenedores, cada servicio posee su propio contenedor y 
orquestado mediante un archivo `docker-compose.yml`.
Es indispensable prefio a la ejecucion del contenedor crear un archivo que contiene las credenciales del sistema. El
archivo debe llevar el nombre de `passwords.json` y debe contener lo siguiente:

```
{
  "forgot_password": "email-recovery-password",
  "smtp_password": "smtp_password",
  "imap_password": "imap_password"
}
```

Por motivos de seguridad se deben completar las credenciales antes mencionadas con sus propias credenciales. De no 
ser posible comunicarse con `fedepacher@gmail.com`.

## Implementacion con docker

Para ejecutar la aplicacion es necesario ejecutar el siguiente comando:

```
docker compose up --build
```

Esto levantara las bases de datos MySQL y MongoDB, ademas ejecutara la API, Ollama el cual descargara el modelo Mistral 
y el servicio de RAG.


## Acceso a la base de datos MySQL

Para acceder a la base de datos MySQL en el contenedor se puede ejecutar el siguiente comando una vez el contenedor 
este corriendo:

```
docker exec -it rag-database-1 bash
```
Para inspeccionar la base de datos debemos entrar a la misma mediante el comando:

```
mysql -u rag-system -p
```
Nos solicitara una contraseña la cual es `rag-system`
Seleccionamos la base de datos creada:

```
use test_sql;
```

Y podremos inspeccionar el contenido de dicha base de datos con el siguiente comando SQL:

```
SELECT * FROM users;
```

<p align=center><img src=resources/assets/query_sql.png height=100 weight=900><p>


## Acceso a la base de datos MongoDB

Para acceder a la base de datos MySQL en el contenedor se puede ejecutar el siguiente comando una vez el contenedor 
este corriendo:

```
docker exec -it rag-mongodb-1 bash
```

Nos conectamos a la base de datos con el siguiente comando:

```
mongosh --host localhost --port 27017 -u mongoadmin -p secret --authenticationDatabase admin
```

Seleccionamos la base de datos creada:

```
use test_nosql
```

Y podremos inspeccionar el contenido de dicha base de datos con el siguiente comando:

```
db.prompts.find().toArray()
```

<p align=center><img src=resources/assets/query_nosql.png height=500 weight=900><p>


# api

DB_NAME: test_sql
DB_USER: rag-system
DB_PASS: rag-system
DB_HOST: localhost
DB_PORT: 3306
ACCESS_TOKEN_EXPIRE_MINUTES: 1440
SECRET_KEY: 6b3d75e968f26f3be3443503efefd761e22d5e173fea95646a3659e673ebb97b
MONGO_HOST: localhost
MONGO_PORT: 27017
MONGO_USER: mongoadmin
MONGO_PASS: secret
MONGO_DB_NAME: test_nosql
      
      

# rag
      
API_URL=http://127.0.0.1:8000
DOCUMENT_LOCATION=resources/files
MONGO_DB_NAME=test_nosql
MONGO_DB_NAME=test_nosql
MONGO_HOST=localhost
MONGO_PASS=secret
MONGO_PORT=27017
MONGO_USER=mongoadmin
PYTHONUNBUFFERED=1

# Ollama server

ollama serve

ollama run mistral