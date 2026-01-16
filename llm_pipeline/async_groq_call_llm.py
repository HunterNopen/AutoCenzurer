import os
import asyncio
from typing import List
from groq import AsyncGroq, RateLimitError

RATE_LIMIT_DELAY = 2.5

async def async_groq_call(prompts: List[str]) -> List[str]:
    client = AsyncGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        timeout=60.0,
    )
    llm_prompts = [{"role": "system", "content": prompt} for prompt in prompts]
    llm_results_labels = []
    try:
        async_result = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=llm_prompts,
            temperature=0.3,
            max_tokens=512,
            stream=False,
            response_format={"type": "json_object"}
        )
        llm_results_labels.append(async_result.choices[0].message.content)
    except Exception as e:
        print(f"Wow... An error {e}")
    finally:
        await client.close()

    return llm_results_labels

async def semaphore_async_groq_call(client: AsyncGroq, prompt: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        await asyncio.sleep(RATE_LIMIT_DELAY)

        try:
            response = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=256,
            stream=False,
            response_format={"type": "json_object"}
        )
            return response.choices[0].message.content
        except RateLimitError as e:
            print(f"!!! HIT RATE LIMIT: {e} !!!")
            await asyncio.sleep(10)
            try:
                response = await client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.3,
                    max_tokens=256,
                    stream=False,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            except Exception as retry_e:
                raise RuntimeError(f"Rate limit retry failed: {retry_e}") from retry_e
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}") from e

async def run_semaphore_groq_call(prompts: List[str]) -> List[str]:
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    semaphore = asyncio.Semaphore(3)

    tasks = [semaphore_async_groq_call(client, prompt, semaphore) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    await client.close()

    return results

async def process_groq_batch_concurrently(prompts: List[str], max_concurrent: int) -> List[str]:
    async with AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"), max_retries=0) as client:
        semaphore = asyncio.Semaphore(max_concurrent) 
        tasks = [semaphore_async_groq_call(client, p, semaphore) for p in prompts]

        return await asyncio.gather(*tasks)
    
def run_groq_batch_concurrently(prompts: List[str], max_concurrent: int = 2) -> List[str]:
    return asyncio.run(process_groq_batch_concurrently(prompts, max_concurrent=max_concurrent))