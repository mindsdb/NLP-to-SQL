# ITG

## Running
1. Clone repo
2. Add the cloned directory to your `PYTHONPATH` (e.g. at the end of .bashrc or .zshrc add `export PYTHONPATH=$PYTHONPATH:/where/you/cloned/itg`)
3. To test it: `python3 tests/test.py`

## Training
1. Install `git-lfs` 
2. Run `git lfs pull`
3. Run `unzip spider.zip && unzip sparc.zip` (don't worry, they are already in the `.gitignore`)

## Ongoing

Try: https://nlpcloud.io/home/playground/code-generation accuracy
Try: https://huggingface.co/mrm8488/t5-base-finetuned-wikiSQL model

## Current insights

Training OpenAI curie (pressumably applies to davinci) seems to indeed be better with more epochs (so it's learning something and seems to behave between with \n separated text than json). Still very bad, 2% hit rate for the best model I obtained (all others are 0%)

## Blocked

- try davinci cortex (requires open ai private beta)
- try fine tuning davinci (requires open ai private beta)
- try fine tuning with > 5 epcohs (require permission from open ai)
- try GPT-J / GPT-Neo (require GCS setup for running Jax / Tensorflow on TPUs, probably will take a bit to get going)