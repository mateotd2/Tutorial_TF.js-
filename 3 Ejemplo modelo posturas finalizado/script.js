

const MODEL_PATH = "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
// Cargamos la imagen del html
const EXAMPLE_IMG = document.getElementById('exampleImg');
let movenet = undefined;

async function loadAndRunModel(){
    // Eston indica a TF.js que intentas cargar un modelo del Hub hosteado por ellos
    movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

    // Con la imagen la transformamos en tensor mediante la funcion tf.browser.fromPixels()
    let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
    console.log(imageTensor.shape);


    let cropStartPoint = [15,170,0];
    let cropSize = [345,345,3];
    let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);


    let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt();
    console.log(resizedTensor.shape);
  
    
  
    let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
    let arrayOutput = await tensorOutput.array();
    console.log(arrayOutput);

}

loadAndRunModel();