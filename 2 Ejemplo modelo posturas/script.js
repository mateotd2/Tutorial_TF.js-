

const MODEL_PATH = "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
let movenet = undefined;

async function loadAndRunModel(){
    // Eston indica a TF.js que intentas cargar un modelo del Hub hosteado por ellos
    movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

    // Dummy valores para comprobar que todo funciona correcto
    // Crea un tensor con forma [1,192,192,3] con todo ceros
    // Al ser todo ceros esto representaria una imagen totalemente en negro (rgb(0,0,0))
    let ejemploEntradaTensor = tf.zeros([1,192,192,3],'int32');

    // Se le pasa los valor de entrada al modelo para intentar predecir la salida
    // La salida; segun la documentacion del modelo; sera de la forma [1,6,56]
    // 1 dimension del lote, el cual es siempre 1
    // 6 numero maximo de instancias detectadas, puede encontrar hasta a 6 personas.
    // 56 se divide en:
        // 3 * 17 elementos donde y_i,x_i,s_i son las coordenadas (x, y) del elemento i con una confianza de s
        // Los restantes 6 elementos [ymin, xmin, ymax, xmax, score] representan la region del cuadro delimitador y su confianza
    let salidaDelTensor = movenet.predict(ejemploEntradaTensor);
    let arraySalida = await salidaDelTensor.array();

    console.log(arraySalida);

}

loadAndRunModel();

