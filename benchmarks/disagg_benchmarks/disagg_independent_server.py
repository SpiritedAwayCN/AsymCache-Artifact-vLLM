import os
import asyncio
import aiohttp
from quart import Quart, jsonify, request, make_response

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                return await response.text()
            else:
                # If the response status is not 200, return the error message
                error_message = await response.text()
                raise Exception(f"Error {response.status}: {error_message}")


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        # Get client request data
        original_request_data = await request.get_json()

        # Prepare prefill request data
        prefill_request = original_request_data.copy()
        prefill_request['max_tokens'] = 1

        # Get environment variables
        vllm_port_prefill = os.environ.get('VLLM_PREFILL_PORT', '8100')
        vllm_port_decode = os.environ.get('VLLM_DECODE_PORT', '8200')

        # Launch prefill and decode tasks
        prefill_task = asyncio.create_task(
            forward_request(f'http://localhost:{vllm_port_prefill}/v1/completions', prefill_request)
        )
        decode_task = asyncio.create_task(
            forward_request(f'http://localhost:{vllm_port_decode}/v1/completions', original_request_data)
        )

        # Wait for both tasks to complete
        prefill_result, decode_result = await asyncio.gather(prefill_task, decode_task)

        # Return the decode result
        response = await make_response(decode_result)
        response.timeout = None
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8000)



