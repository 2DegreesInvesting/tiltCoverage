# Classifying ecoinvent activity and products

Aim: assign ecoinvent activities and prooducts to companies retrieved from different data resources. We focus on the Orbis as the data resource.

## Experiments

Data set: We have a data set of companies in Orbis that has been assigned `activity_uuid_product_uuid`.

Task: assign the appropricate ecoinvent `activity_uuid_product_uuid` to companies retrieved from Orbis. Each company in the data set has *at least* one `activity_uuid_product_uuid` assigned, therefore this is a multilabel classification task.

Evaluation: As a multilabel classification task, we consider a prediction correct if the predicted label is one of the ground truth labels for each data input.

### Retrieval

There are 3171 unique  `activity_uuid_product_uuid` in `20231121_mapper_ep_ei.csv`. At first glance, is simply too many classes to apply generative AI. We can classify by means of retrieval. With activity and product descriptions of the `activity_uuid_product_uuid` as the documents and the company descriptions as the query, we retrieve the most appropriate `activity_uuid_product_uuid` by computing their semantic similarity as the relevance metric. Below are experiments we perform using `haystack` retrievers.

**Initial set up**

* Using the `haystack` framework: `FAISSDocumentStore` as the document store and `EmbeddingRetriever` and the retriever.
* Documents in the document store:
    * Content: Concatenation (using `;`) of the descriptions of the ecoinvent activity and product that correspond to the `activity_uuid` and `product_uuid` of `activity_uuid_product_uuid`, i.e. `<activity descr>; <product descr>`
    * ID: `activity_uuid_product_uuid`
* Using cosine similarity


**Experiment 1**

How does the huggingface model fare against OpenAI?

*Set up*

* Initial set up
* Retriever embedder model:
    * `sentence-transformers/all-MiniLM-L6-v2` (embedding size 384) from huggingface
    * `text-embedding-3-small` (embedding size ???) from OpenAI
* Query construction: Concatenation (using `;`) of the descriptions of the NACE and NAICS codes, and the product and services (if available), i.e. `<NAICS code descr>; <NACE code descr>;<product_and_services>`

*Results - accuracy*

* huggingface model: 1.17%
* OpenAI model: %

**Experiment 2**

We want to include the trade description, which are in Dutch: does using trade description with multilingual models add to the accuracy of the retrieval?

*Set up*

* Initial set up
* Retriever embedder models:
    * `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (embedding size 384) from hf
    * `FacebookAI/xlm-roberta-base` (embedding size 384) from hf
    * `text-embedding-3-small` (embedding size ???) from OpenAI 
* Query construction: Addition of the trade description to the original query construction.

*Results - accuracy*

* `paraphrase-multilingual-MiniLM-L12-v2`:
* `xlm-roberta-base`: 

*Results - observation*

* `xlm-roberta-base` is quite large and takes a long time to evaluate even the validation set - over 4 hours

**Experiment 3**

Does fine-tuning the retriever model improve the performance?

*Set up*

* ...

*Results - accuracy*

* ...

### Step-wise retrieval

As there are many combinations of ecoinvent activity and product possible, the differences between certain `acitivity_uuid_product_uuid` may be too subtle. In the following experiments, we see if first narrowing down the activity and then searching within the subset of products results in improved prediction performance.

**Initial set up**
