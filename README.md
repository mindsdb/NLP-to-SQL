# ITG

## Running
1. Clone repo
2. Add the cloned directory to your `PYTHONPATH` (e.g. at the end of .bashrc or .zshrc add `export PYTHONPATH=$PYTHONPATH:/where/you/cloned/itg`)
3. To test it: `python3 tests/test.py`

## Training
1. Install `git-lfs` 
2. Run `git lfs pull`
3. Run `unzip spider.zip && unzip sparc.zip` (don't worry, they are already in the `.gitignore`)


## Current insights

Untrained, OpenAI gives better results than NLP-Cloud.

Both OpenAI and NLP-Cloud are very good provided:
1. Few shot learning (i.e. 2-3 valid examples then the "query")
2. Small inputs (they can't deal with big inputs)
3. Known approximate output size
Fine tuning OpenAI models (tried ~14 different hyperparameter and data format combos), however, *always* just makes them worst, they go from "slightly" wrong to "complete nonsense".
This more or less seems to fit the "stochastic parrot" hypothesis, where these models are still not capable of generalization :/

Hugging-face models with training showcase somewhat more promising results than both OpenAI and NLP Cloud. But training seems to easily break the model (e.g. it starts always predicting padding tokens). Training the models for seq2seq predictions "properly" is hard, official docs on the topic are few and mainly outdated, examples are incomplete and other tutorials are just blog-spam.

Setting up and fine-tuning GPT-J also seems non trivial (I assume it generalizes to neo too), I failed even trying to get the env working.

## Blocked

- try davinci cortex (requires open ai private beta)
- try fine tuning davinci (requires open ai private beta)
- try fine tuning with > 5 epcohs (require permission from open ai)
- try GPT-J / GPT-Neo (require GCS setup for running Jax / Tensorflow on TPUs, probably will take a bit to get going)
- try using hugging-face's new trainer interface to tune a large~ish model from scratch (required feedback from community since I can't figure it out and the docs are basically missing)
- try fine tuning nlp cloud (requires blind-spend of 500$ or reply from them, since they mentioned contacting them if we want to try fine-tuning)

## Path Forward

- The best path forward to getting reasonable "simple" sql seems to be trying to custom-train a hugging-face model ala T5-11B on large GPUs with SPDIDER + SPARC + WIKITEXT + maybe a few more. Using the current processing techniques for minimizing input size to make it suitable for transformers + select only the easiest queries.
- To deal with large database sizes it seems reasonable to train a "Summarization" model that takes metadata about the schema + the user text query and figures out the relevant tables for it, rather than the sql-query. This would help overcomes the tiny-inputs limitation.
- To gain the final bits of improvement it probably makes sense to add heuristics around query-evaluation (i.e. once a model gives an output try to "extract" a valid sql query from it which adheres to the known columns + tables in the current database => if you can't, ask for another prediction and loop this => If nothing valid is generated, report and error). Given current performance + poor results on ladderboard for text to SQL we probably need all the help we can get here, and focusing on simple queries + evaluation heuristics might get us to the necessary accuracy.

## Open threads

Mainly useful in case someone else takes this over.
Currently I have some questions I'm waiting answers to laid out in these threads:
- https://stackoverflow.com/questions/70358307/how-to-train-hugging-face-models-with-an-lm-head
- https://stackoverflow.com/questions/70374173/how-to-avoid-huggingface-t5-based-seq-to-seq-suddenly-reaching-a-loss-of-nan-a
- https://discuss.huggingface.co/t/training-t5-based-seq-to-seq-suddenly-reaches-loss-of-nan-and-starts-predicting-only-pad/12884/2
- https://discuss.huggingface.co/t/extremely-confusing-or-non-existent-documentation-about-the-seq2seq-trainer/12880
As well as an email thread with nlpcloud about fine-tuning (will add it here later)