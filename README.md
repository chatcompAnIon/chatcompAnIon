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
**compAnIon-v1.0** is a transformer-based large language model (LLM) that was trained for child grooming text classification in gaming chat room environments. **compAnIon-v1.0** is a lightweight model consisting of only 110,479,516 total parameters designed to deliver classification decisions in milliseconds within chat room dialogues. Our model was trained on non-grooming chat data from several sources including IRC Logs, Omegle, and the Chit Chats dataset. Our grooming chat instances were extracted from the Perverted Justice dataset, representing ~3k conversation histories of convicted child groomers with their predators. 

Predicting child grooming is  incredibly difficult accentuated by the huge class imbalance given the rarity of publicly available grooming chat data and the omnipresence of nongrooming chat data. As such our dataset consistents of ~3% positive classes. Through a combination of up/downsampling, we arrived at a final training data mix consisting of 25% positive grooming classes. To help combat the remaining class imbalance 
we relied on the [Binary Focal Crossentropy](https://arxiv.org/abs/1708.02002v2) loss function with customized gamma, aimed at penalizing the model for overfitting on the easier-to-predict class. 

The model was initially designed to be trained on the chat texts in addition to new features engineered from linguistic analysis within the field of child grooming. As child grooming (especially in chats) often follows a [lifecycle of stages](https://safechild.org/understanding-grooming/) such as (building trust, isolating the child, etc.) we found such features rather simple to manually extract through various techniques such as through Bag of Words types models. 

The chat texts and new features were fed into the *bert-base-uncased* encoder to build embedding representations of which the pooler output with a dimension of 768 was extracted for each conversation. This embedding representation was fed into dense neural network layers to produce an ultimate binary classification. However, after exhaustive ablation studies, we discovered that the inclusion of 1D convolution layers on top of our text embeddings acted as a much more effective automated way to extract features. As such **compAnIon-v1.0** relies solely on the convolutional filters to act as feature extractors before feeding into the dense neural network layers.


| :---         |     :---:      |   
| Instance Size  | g5.4xlarge     | 
| GPU    | 1       | 
| GPU Memory (GiBs)    | 24     | 
| vCPUs    | 16       | 
| Memory (GiB)    | 64      | 
| Instance Storage (GB)    | 1 x 600 NVMe SSD       | 
| Network Bandwidth (Gbps)    | 25       | 
| EBS Bandwidth (Gbps)    | 8       | 




### Prerequisites

In order to run compAnIon-v1.0, the following installs are required: 

  ```python
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
