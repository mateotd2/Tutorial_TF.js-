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

const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}


const MODEL_PATH = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';

let model = undefined;



async function loadModel() {

  model = await tf.loadLayersModel(MODEL_PATH);

  model.summary();

  

  // Create a batch of 1.

  const input = tf.tensor2d([[870]]);

  

  // Create a batch of 3

  const inputBatch = tf.tensor2d([[500], [1100], [970]]);


  // Actually make the predictions for each batch.

  const result = model.predict(input);

  const resultBatch = model.predict(inputBatch);

  

  // Print results to console.

  result.print();  // Or use .arraySync() to get results back as array.

  resultBatch.print(); // Or use .arraySync() to get results back as array.

  

  input.dispose();

  inputBatch.dispose();

  result.dispose();

  resultBatch.dispose();

  model.dispose();

}


loadModel();