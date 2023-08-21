import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
import joblib
import time
import pickle
import tensorflow as tf
from transformers import (
    pipeline
)

app = FastAPI()


@app.get("/")
async def root():
    """root"""
    return {"message": "Nothing to see here"}


@app.get("/gpt2_model")
async def use_gpt2_model(model_name, prompts, task="text-generation", framework="tf", max_tokens=200):
    """endpoint to use gpt2 model"""
    try:
        specified_model, tokenizer = load_specified_model(model_name)
        start_time = time.time()

        # Split data for parallel processing
        num_cores = joblib.cpu_count()
        batch_size = len(prompts) // num_cores
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

        # Parallel processing
        results = joblib.Parallel(n_jobs=num_cores)(
            joblib.delayed(run_gpt2_model)(specified_model, tokenizer, prompt,
                                           task, framework, max_tokens) for prompt in batches
        )

        # Combine results
        combined_results = []
        for result_batch in results:
            combined_results.extend(result_batch)

        end_time = time.time()
        elapsed_time = end_time - start_time

        tps = len(prompts) / elapsed_time

        return {'results': combined_results, 'tps': tps}

    except BaseException as error:
        return f"Something went wrong: {error}"


@app.get("/llama_model")
async def use_llama_model(model_name, prompts, task="text-generation", framework="pt", max_tokens=200):
    """endpoint to use llama model"""
    try:
        specified_model, tokenizer = load_specified_model(model_name)
        start_time = time.time()

        # Split data for parallel processing
        num_cores = joblib.cpu_count()
        batch_size = len(prompts) // num_cores
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

        # Parallel processing
        results = joblib.Parallel(n_jobs=num_cores)(
            joblib.delayed(run_llama_model)(specified_model, tokenizer, prompt,
                                            task, framework, max_tokens) for prompt in batches
        )

        # Combine results
        combined_results = []
        for result_batch in results:
            combined_results.extend(result_batch)

        end_time = time.time()
        elapsed_time = end_time - start_time

        tps = len(prompts) / elapsed_time

        return {'results': combined_results, 'tps': tps}

    except BaseException as error:
        return f"Something went wrong: {error}"


def load_specified_model(model_name):
    # Load saved model
    DEFAULT_FUNCTION_KEY = "serving_default"
    loaded_model = tf.saved_model.load(model_name)
    inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]
    print(loaded_model)
    print(inference_func)

    # Load saved tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(tokenizer)

    return loaded_model, tokenizer


def run_gpt2_model(model, tokenizer, prompt,  task="text-generation", framework="tf", max_tokens=200):
    # Enable XLA for TensorFlow
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({"xla_gpu": True})

    # Initialize pipeline with XLA-enabled TensorFlow
    pipe = pipeline(
      task=task,
      model=model,
      tokenizer=tokenizer,
      max_length=max_tokens,
      framework=framework,  # Set the framework
    )

    # Run text generation pipeline with our next model
    result = pipe(f"{prompt}")
    return result[0]['generated_text']


def run_llama_model(model, tokenizer, prompt,  task="text-generation", framework="tf", max_tokens=200):
    # Run text generation pipeline with llama model
    pipe = pipeline(task=task, model=model, tokenizer=tokenizer, max_length=max_tokens)

    # Run text generation pipeline with our next model
    result = pipe(f"{prompt}")
    return result[0]['generated_text']


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
