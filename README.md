# chatcompAnIon
<br />
<div align="center">
  <a href="https://https://github.com/chatcompAnIon/chatcompAnIon">
    <img src="images/Chat Companion Logo.png" alt="Logo" width="400" height="400">
  </a>

  <h3 align="center">Welcome to Chat CompAnIon's repo!</h3>

  <p align="center">
    <br />
    <a href="https://cleemazzulla.github.io/chatcompAnIon/"><strong>Visit our Webpage»</strong></a>
    <br />
    <br />
  </p>
</div>

<!-- ADD IN LATER TABLE OF CONTENTS -->

<!-- GETTING STARTED -->

## Getting Started

To help combat what has been deemed an as *AN INDUSTRY WITHOUT AN ANSWER*, chat compAnIon is making the first model **compAnIon-v1.0** publicly available. To help facilitate reproducability we have made our model available via Hugging Face: [chatcompanion/compAnIon-v1.0](https://huggingface.co/chatcompanion/compAnIon-v1.0)

### Model Overview
**compAnIon-v1.0** is a transformer-based large language model (LLM) that was trained for child grooming text classification in gaming chat room environments. The model is a lightweight model consisting of only XYZM total parameters designed to deliver classification decision in miliseconds within 


### Prerequisites

In order to run compAnIon-v1.0, the following installs are required: 
* npm
  ```sh
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.common import *
from pyspark.sql.functions import *
from pyspark.sql.functions import lit
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler, Imputer, OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.pandas as ps

import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

### Example run on a GPU
 ```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", trust_remote_code=True, token="hf_YOUR_TOKEN")
model = AutoModelForCausalLM.from_pretrained("databricks/dbrx-instruct", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, token="hf_YOUR_TOKEN")

input_text = "What does it take to build a great LLM?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))

   ```

<!-- CONTACT -->
## Contact

* [Courtney Mazzulla](https://www.linkedin.com/in/courtney-l-mazzulla/) - cleemazzulla@berkeley.edu
* [Julian Rippert](https://www.linkedin.com/in/julianrippert/) - jrippert@berkeley.edu
* [Sunny Shin](https://www.linkedin.com/in/sunnyshin1/) - sunnyshin@berkeley.edu
* [Raymond Tang](https://www.linkedin.com/in/raymond-tang-0807aa1/) - raymond.tang@berkeley.edu
* [Leon Gutierrez](https://www.linkedin.com/in/leongutierrez29/) - leonrafael29@berkeley.edu
* [Karsyn Lee](https://www.linkedin.com/in/karsynlee/) - karsyn@berkeley.edu

[Project Website](https://cleemazzulla.github.io/chatcompAnIon/)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project was developed as a part of UC Berkeley's Master of Information and Data Science Capstone. We would like to thank our Capstone advisors Joyce Shen and Korin Reid for their extensive guidance and continued support. We invite to visit out cohort's projects as well: [MIDS Capstone Projects: Spring 2024](https://www.ischool.berkeley.edu/programs/mids/capstone/2024a-spring)
