/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js";

// Grab a reference to the MNIST input values (pixel data).
const INPUTS = TRAINING_DATA.inputs;

// Grab reference to the MNIST output values.
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays to remove any order, but do so in the same way so
// inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);


function normalize(tensor, min, max) {

  const result = tf.tidy(function() {

    const MIN_VALUES = tf.scalar(min);

    const MAX_VALUES = tf.scalar(max);


    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);


    return NORMALIZED_VALUES;

  });

  return result;

}


// Input feature Array is 2 dimensional.

const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

// Output feature Array is 1 dimensional.
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// Now actually create and define model architecture.

const model = tf.sequential();


model.add(tf.layers.conv2d({

  inputShape: [28, 28, 1],

  filters: 16,

  kernelSize: 3, // Square Filter of 3 by 3. Could also specify rectangle eg [2, 3].

  strides: 1,

  padding: 'same',

  activation: 'relu'  

}));


model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

model.add(tf.layers.conv2d({

  filters: 32,

  kernelSize: 3,

  strides: 1,

  padding: 'same',

  activation: 'relu'

}));


model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

model.add(tf.layers.flatten());


model.add(tf.layers.dense({units: 128, activation: 'relu'}));


model.add(tf.layers.dense({units: 10, activation: 'softmax'}));


model.summary();

train();

async function train() { 

  model.compile({

    optimizer: 'adam', // Adam changes the learning rate over time which is useful.

    loss: 'categoricalCrossentropy', // As this is a classification problem, dont use MSE.

    metrics: ['accuracy'] 

  });


  const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]);

  let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {

    shuffle: true,        // Ensure data is shuffled again before using each time.

    validationSplit: 0.15,  

    epochs: 30,           // Go over the data 30 times!

    batchSize: 256,

    callbacks: {onEpochEnd: logProgress}

  });

  RESHAPED_INPUTS.dispose();

  OUTPUTS_TENSOR.dispose();

  INPUTS_TENSOR.dispose();

  evaluate();

}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

// Map output index to label.

const LOOKUP = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

function evaluate() {

  const OFFSET = Math.floor((Math.random() * INPUTS.length));


  let answer = tf.tidy(function() {

    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255);

    let output = model.predict(newInput.reshape([1, 28, 28, 1]));

    output.print();

    return output.squeeze().argMax();    

  });

  answer.array().then(function(index) {

    PREDICTION_ELEMENT.innerText = LOOKUP[index];

    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();

    drawImage(INPUTS[OFFSET]);

  });

}

const CANVAS = document.getElementById("canvas");

function drawImage(digit) {
  digit = tf.tensor(digit, [28, 28]).div(255);
  tf.browser.toPixels(digit, CANVAS);

  // Perform a new classification after a certain interval.
  setTimeout(evaluate, interval);
}

var interval = 2000;

const RANGER = document.getElementById('ranger');

const DOM_SPEED = document.getElementById('domSpeed');


// When user drags slider update interval.

RANGER.addEventListener('input', function(e) {

  interval = this.value;

  DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms';

});