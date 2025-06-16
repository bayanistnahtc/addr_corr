import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tenacity import (
    RetryError, 
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type
)
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException


logger = logging.getLogger(__name__)

# Define exceptions that should trigger retries
TRITON_RETRYABLE_EXCEPTIONS = (
    InferenceServerException,
    ConnectionRefusedError,
    TimeoutError,
    asyncio.TimeoutError
)


@dataclass
class LLMSamplingParams:
    """
    Dataclass for LLM sampling parameters.
    
    Attributes:
        temperature: Controls randomness in generation. Lower values make output more deterministic.
        top_p: Nucleus sampling parameter. Only tokens with cumulative probability <= top_p are considered.
        top_k: Only the top k tokens are considered at each step.
        max_tokens: Maximum number of tokens to generate.
        stop: List of stop sequences.
        presence_penalty: Penalty for tokens that have appeared in the text.
        frequency_penalty: Penalty for tokens based on their frequency.
    """
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = -1  # -1 means no limit
    max_tokens: int = 64
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class TritonManager:
    """
    Manages connection and inference with Triton Inference Server.

    Provides methods for checking server status, model readiness,
    and performing async batch inference with proper error handling and retries.

    Inspired by concepts from:
    https://github.com/triton-inference-server/vllm_backend/blob/main/samples/client.py
    """

    # Default names for model inputs and outputs
    TEXT_INPUT_NAME: str = "text_input"
    TEXT_OUTPUT_NAME: str = "text_output"
    SAMPLING_PARAMS_INPUT_NAME: str = "sampling_parameters"
    STREAM_INPUT_NAME: str = "stream"
    EXCLUDE_INPUT_OUTPUT_NAME: str = "exclude_input_in_output"

    def __init__(
            self, 
            triton_url: str, 
            model_name: str,
            model_version: str = "1",
            request_timeout: float = 30.0,
            stream_timeout: Optional[float] = None,
            verbose: bool = False,
            sampling_params: Optional[LLMSamplingParams] = None,
            exclude_input_in_output: bool = True
        ):
        """
        Initialize the TritonManager.

        Args:
            triton_url: URL of the Triton Inference Server (host:port).
            model_name: Name of the model registered in Triton.
            model_version: Specific model version. Defaults to "1".
            request_timeout: Timeout in seconds for inference requests. Defaults to 30.0.
            stream_timeout: Timeout for streaming operations. Defaults to None.
            verbose: Enable verbose logging from the Triton client. Defaults to False.
            sampling_params: LLM sampling parameters. If None, uses defaults.
            exclude_input_in_output: Whether to exclude input prompt from output. Defaults to True.
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.request_timeout = request_timeout
        self.stream_timeout = stream_timeout
        self.verbose = verbose
        self.exclude_input_in_output = exclude_input_in_output
        
        # Set default sampling parameters if not provided
        self.sampling_params = sampling_params or LLMSamplingParams()
        
        # Connection state
        self._client: Optional[grpcclient.InferenceServerClient] = None
        self._metadata_cache: Optional[Dict[str, Any]] = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(TRITON_RETRYABLE_EXCEPTIONS),
        reraise=True
    ) 
    async def check_server_live(self) -> bool:
        """
        Check if the Triton server is live.

        Returns:
            True if the server is live, False otherwise.
        
        Raises:
            InferenceServerException: For specific Triton errors.
            Exception: For unexpected errors.
        """
        client = await self._get_client()
        try:
            return await client.is_server_live()
        except InferenceServerException as e:
            logger.warning(f"Triton server liveness check failed: {e}")
            raise 
        except Exception as e:
            logger.error(f"Unhandled error during Triton liveness check: {e}", exc_info=True)
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(TRITON_RETRYABLE_EXCEPTIONS),
        reraise=True
    )       
    async def check_model_ready(self) -> bool:
        """
        Check if the model is ready on the Triton server.

        Fetches and caches model metadata if not already cached.

        Returns:
            True if the model is ready, False otherwise.

        Raises:
            InferenceServerException: For specific Triton errors.
            Exception: For unexpected errors.
        """
        client = await self._get_client()
        
        try:
            is_ready = await client.is_model_ready(
                model_name=self.model_name,
                model_version=self.model_version
            )

            if not is_ready:
                logger.warning(f"Model '{self.model_name}' is not ready on Triton server.")
                return False

            # Fetch metadata if not already cached
            if self._metadata_cache is None:
                await self._fetch_model_metadata()

            return is_ready
        except InferenceServerException as e:
            if "not found" in str(e).lower():
                 logger.error(f"Triton model '{self.model_name}' version '{self.model_version}' not found on server {self.triton_url}.")
                 return False
            logger.warning(f"Triton model '{self.model_name}' readiness check failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unhandled error during Triton readiness check: {e}", exc_info=True)
            return False

    async def close(self) -> None:
        """
        Close the Triton client connection.

        Should be called when the manager is no longer needed to free resources.
        """
        if self._client:
            try:
                await self._client.close()
                logger.info("Triton client closed successfully")
                self._client = None
            except Exception as e:
                logger.error(f"Error closing Triton client: {e}", exc_info=True)
    
    async def predict(self, text: str) -> str:
        """
        Send a single text input to the model and retrieve the result.

        Args:
            text: The input text to process.

        Returns:
            The processed text from the model.

        Raises:
            InferenceServerException: On server communication errors.
        """
        results = await self.predict_batch([text])
        return results[0] if results else ""

    async def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Send a batch of text inputs to the Triton server for inference.

        Args:
            texts (List[str]): A list of text inputs to be processed by the model.

        Returns:
            List[str]: A list of processed strings returned by the model.

        Raises:
            InferenceServerException: On server communication errors.
            ValueError: If an empty input list is provided.
        """
        if not texts:
            logger.debug("predict_batch called with empty input list")
            return []

        try:
            # Process all texts through async streaming
            results = await self._process_batch_streaming(texts)
            return results
            
        except InferenceServerException as e:
            logger.error(f"Triton inference failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Triton prediction: {e}", exc_info=True)
            raise

    async def _process_batch_streaming(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts using vLLM async streaming.
        
        Args:
            texts: List of input texts to process.
            
        Returns:
            List of generated text outputs.
        """
        client = await self._get_client()
        results_dict = {}
        
        try:
            # Create async request iterator
            async def request_iterator():
                for i, text in enumerate(texts):
                    request_id = str(i)
                    results_dict[request_id] = []
                    yield self._create_llm_request(text, request_id, streaming=False)
            
            # Start streaming inference
            response_iterator = client.stream_infer(
                inputs_iterator=request_iterator(),
                stream_timeout=self.stream_timeout
            )
            
            # Process streaming responses
            async for response in response_iterator:
                result, error = response
                if error:
                    logger.error(f"Encountered error while processing: {error}")
                    raise InferenceServerException(f"Streaming inference error: {error}")
                else:
                    # Extract text output
                    output = result.as_numpy(self.TEXT_OUTPUT_NAME)
                    request_id = result.get_response().id
                    
                    # Store results for this request
                    for output_text in output:
                        if isinstance(output_text, bytes):
                            decoded_text = output_text.decode("utf-8")
                        else:
                            decoded_text = str(output_text)
                        results_dict[request_id].append(decoded_text)
            
            # Reconstruct results in original order
            final_results = []
            for i in range(len(texts)):
                request_id = str(i)
                if request_id in results_dict and results_dict[request_id]:
                    # Join all output chunks for this request
                    final_results.append("".join(results_dict[request_id]))
                else:
                    final_results.append("")
                    
            return final_results
            
        except Exception as e:
            logger.error(f"Error in batch streaming processing: {e}", exc_info=True)
            raise

    def _create_llm_request(
        self, 
        text: str, 
        request_id: str, 
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Create a request dictionary compatible with Triton model.
        
        Args:
            text: Input text prompt.
            request_id: Unique identifier for this request.
            streaming: Whether to enable streaming mode.
            
        Returns:
            Dictionary containing the request configuration.
        """
        inputs = []
        
        # Text input
        prompt_data = np.array([text.encode("utf-8")], dtype=np.object_)
        text_input = grpcclient.InferInput(self.TEXT_INPUT_NAME, [1], "BYTES")
        text_input.set_data_from_numpy(prompt_data)
        inputs.append(text_input)
        
        # Stream parameter
        stream_data = np.array([streaming], dtype=bool)
        stream_input = grpcclient.InferInput(self.STREAM_INPUT_NAME, [1], "BOOL")
        stream_input.set_data_from_numpy(stream_data)
        inputs.append(stream_input)
        
        # Sampling parameters as JSON
        sampling_params_dict = {
            "temperature": str(self.sampling_params.temperature),
            "top_p": str(self.sampling_params.top_p),
            "max_tokens": str(self.sampling_params.max_tokens),
        }
        
        # Add optional parameters if they differ from defaults
        if self.sampling_params.top_k > 0:
            sampling_params_dict["top_k"] = str(self.sampling_params.top_k)
        if self.sampling_params.stop:
            sampling_params_dict["stop"] = self.sampling_params.stop
        if self.sampling_params.presence_penalty != 0.0:
            sampling_params_dict["presence_penalty"] = str(self.sampling_params.presence_penalty)
        if self.sampling_params.frequency_penalty != 0.0:
            sampling_params_dict["frequency_penalty"] = str(self.sampling_params.frequency_penalty)
            
        sampling_params_data = np.array(
            [json.dumps(sampling_params_dict).encode("utf-8")], 
            dtype=np.object_
        )
        sampling_input = grpcclient.InferInput(self.SAMPLING_PARAMS_INPUT_NAME, [1], "BYTES")
        sampling_input.set_data_from_numpy(sampling_params_data)
        inputs.append(sampling_input)
        
        # Exclude input in output parameter
        exclude_input_data = np.array([self.exclude_input_in_output], dtype=bool)
        exclude_input = grpcclient.InferInput(self.EXCLUDE_INPUT_OUTPUT_NAME, [1], "BOOL")
        exclude_input.set_data_from_numpy(exclude_input_data)
        inputs.append(exclude_input)
        
        # Requested outputs
        outputs = [grpcclient.InferRequestedOutput(self.TEXT_OUTPUT_NAME)]
        
        return {
            "model_name": self.model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": request_id,
            "parameters": sampling_params_dict,
        }

    async def _fetch_model_metadata(self) -> Dict[str, Any]:
        """
        Fetch and cache model metadata from the Triton server.

        Returns:
            Dict[str, Any]: Dictionary containing model metadata.

        Raises:
            RuntimeError: If metadata cannot be fetched.
        """
        if self._metadata_cache:
            return self._metadata_cache

        try:
            client = await self._get_client()
            metadata = await client.get_model_metadata(
                model_name=self.model_name,
                model_version=self.model_version
            )

            self._metadata_cache = {
                "inputs": {inp.name: inp for inp in metadata.inputs},
                "outputs": {out.name: out for out in metadata.outputs},
            }

            logger.info(f"Model metadata fetched for {self.model_name}")
            logger.debug(f"Available inputs: {list(self._metadata_cache['inputs'].keys())}")
            logger.debug(f"Available outputs: {list(self._metadata_cache['outputs'].keys())}")
            
            return self._metadata_cache

        except InferenceServerException as e:
            logger.error(f"Failed to fetch model metadata: {e}")
            raise RuntimeError(f"Failed to fetch model metadata: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching model metadata: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error fetching model metadata: {e}")

    async def _get_client(self) -> grpcclient.InferenceServerClient:
        """
        Get or initialize the Triton async gRPC client.

        Returns:
            InferenceServerClient: Initialized async InferenceServerClient instance.
        
        Raises:
            ConnectionError: If the client cannot be created.
        """
        if self._client is None:
            logger.info(f"Connecting to Triton server at {self.triton_url}...")
            try:
                self._client = grpcclient.InferenceServerClient(
                    url=self.triton_url,
                    verbose=self.verbose
                )
                logger.info(f"Connected to Triton server at {self.triton_url}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Triton server: {e}", exc_info=True)
                raise ConnectionError(f"Failed to connect to Triton server at {self.triton_url}: {e}")

        return self._client