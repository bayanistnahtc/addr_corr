import logging
from typing import Dict, List, Optional

from schemas.v1.requests import PredictRequest
from schemas.v1.responses import PredictResponse
from core.model_manager import RetryError, TritonManager, LLMSamplingParams


logger = logging.getLogger(__name__)


class AddressCorrectionService:
    """
    Service class for performing address correction using a language model via Triton Inference Server with vLLM backend.
    """
    def __init__(
            self, 
            triton_manager: TritonManager,
            max_batch_size: int = 4,
            input_format: str = "",
            max_input_chars: int = 1000,
            sampling_params: Optional[LLMSamplingParams] = None
            ):
        """
        Initialize the AddressCorrectionService.

        Args:
            triton_manager (TritonManager): An instance of the TritonManager configured 
                for the address correction model with vLLM backend.
            max_batch_size(int): Maximum inputs in batch. Defaults to 4.
            input_format (str): Format string for preparing model inputs.
                Defaults to "geoparser: {geoparser_address} . old: {old_address}</s>".
            max_input_chars (int): Maximum length of input values in chars. Defaults to 1000.
            sampling_params (VLLMSamplingParams, optional): vLLM sampling parameters.
                If provided, will override the triton_manager's default parameters.
        """
        self.triton_manager = triton_manager
        self.max_batch_size = max_batch_size
        if input_format:
            self.input_format = input_format
        else:
            self.input_format = "geoparser: {geoparser_address} . old: {old_address}</s>"
        self.max_input_chars = max_input_chars
        
        # Update sampling parameters if provided
        if sampling_params:
            self.triton_manager.sampling_params = sampling_params

    async def correct_single(self, request: PredictRequest) -> PredictResponse:
        """
        Correct a single address asynchronously.

        Args:
            request (PredictRequest): The address input data.
        
        Returns:
            PredictResponse: A response object containing the corrected address.
        """
        logger.debug(f"Processing single address correction request: {request.request_id}")

        try:
            input_text = self._prepare_model_input(request)
            corrected_address = await self.triton_manager.predict(input_text)

            if not corrected_address:
                logger.warning(f"Empty correction result for request ID: {request.request_id}")
                return PredictResponse(
                    id=request.request_id,
                    corrected_address=None,
                    error_message="No correction result returned from model."
                    )

            return PredictResponse(
                id=request.request_id,
                corrected_address=corrected_address,
                error_message=None)
                
        except RetryError as e:
            logger.error(
                f"Triton Inference failed after retries for request ID {request.request_id}: {e}",
                exc_info=True
            )
            return PredictResponse(
                    id=request.request_id,
                    corrected_address=None,
                    error_message=f"Inference failed after retries: {type(e.__cause__).__name__}"
            )
        except Exception as e:
            logger.error(
                f"Error processing request ID {request.request_id}: {e}",
                exc_info=True
            )
            return PredictResponse(
                    id=request.request_id,
                    corrected_address=None,
                    error_message=f"Processing error: {type(e).__name__} - {str(e)}"
            )

    async def correct_batch(self, requests: List[PredictRequest]) -> List[PredictResponse]:
        """
        Correct a batch of addresses asynchronously.

        Args:
            requests (List[PredictRequest]): A list of address input objects.
        
        Returns:
            List[PredictResponse]: A list of response objects.
        """

        if not requests:
            logger.debug("correct_batch called with empty request list.")
            return []
        
        batch_size = len(requests)

        if batch_size > self.max_batch_size:
            logger.info(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}. Splitting.")
            results = []
            # Process batches sequentially to avoid overwhelming the server
            for i in range(0, batch_size, self.max_batch_size):
                batch_requests = requests[i:i + self.max_batch_size]
                batch_results = await self.correct_batch(batch_requests)
                results.extend(batch_results)
            return results

        request_ids = [request.request_id for request in requests]
        
        logger.debug(f"Processing batch correction for {batch_size} addresses")

        # Prepare inputs for model
        input_texts: List[str] = []
        failed_indices: Dict[int, str] = {}

        for i, request in enumerate(requests):
            try:
                input_texts.append(self._prepare_model_input(request))
            except Exception as e:
                logger.error(
                    f"Error preparing input for request ID: {request.request_id}: {e}",
                    exc_info=True
                )
                failed_indices[i] = f"Input preparation error: {str(e)}"
                input_texts.append("")  # Placeholder for failed input

        # Call triton async
        corrected_addresses: List[Optional[str]] = [None] * batch_size

        # Filter out failed inputs for inference
        valid_indices = [i for i in range(batch_size) if i not in failed_indices]
        valid_texts = [input_texts[i] for i in valid_indices]

        if valid_texts:
            try:
                # Call async batch prediction
                results = await self.triton_manager.predict_batch(valid_texts)
                
                if results is None:
                    logger.error("Triton returned None for batch prediction")
                    for i in valid_indices:
                        failed_indices[i] = "Model returned no results"
                elif len(results) != len(valid_texts):
                    logger.error(f"Result count mismatch. Expected {len(valid_texts)}, got {len(results)} from inference service")
                    for i in valid_indices:
                        failed_indices[i] = "Result count mismatch from inference service"
                else:
                    # Map results back to original positions
                    for idx, result_idx in enumerate(valid_indices):
                        corrected_addresses[result_idx] = results[idx]

            except RetryError as e:
                logger.error(f"Triton inference failed after retries for batch: {e}", exc_info=True)
                batch_error_message = f"Internal error: Inference failed after retries: {type(e.__cause__).__name__}"
                for i in valid_indices:
                    failed_indices[i] = batch_error_message
            except Exception as e:
                logger.error(f"Unhandled exception during Triton predictions for batch: {e}", exc_info=True)
                batch_error_message = f"Internal inference error: {type(e).__name__}"
                for i in valid_indices:
                    failed_indices[i] = batch_error_message
        
        # Construct responses
        results: List[PredictResponse] = []
        for i, request_id in enumerate(request_ids):
            error_message = failed_indices.get(i)
            results.append(
                PredictResponse(
                    id=request_id,
                    corrected_address=corrected_addresses[i],
                    error_message=error_message
                )
            )
        
        successful_count = sum(1 for r in results if r.error_message is None)
        failed_count = batch_size - successful_count
        logger.info(
            f"Finished batch correction for {batch_size} requests. Success: {successful_count}, Failed: {failed_count}"
        )
        return results

    def _prepare_model_input(self, request: PredictRequest) -> str:
        """
        Prepare the input string based on the request data.
        
        Args:
            request (PredictRequest): The input request object.

        Returns: 
            str: The prepared input string formatted for the model.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if not request.geoparser_address:
            raise ValueError("Missing required field: geoparser_address")
            
        # Handle input length truncation
        geoparser_addr = request.geoparser_address or ""
        old_addr = request.old_address or ""
        
        if len(geoparser_addr + old_addr) > self.max_input_chars:
            logger.warning(
                f"The input value (geoparser_address + old_address) exceeds maximum length ({self.max_input_chars} chars). "
                "Truncating input values."
            )
            # Truncate proportionally to preserve both inputs
            total_len = len(geoparser_addr + old_addr)
            geoparser_ratio = len(geoparser_addr) / total_len
            old_ratio = len(old_addr) / total_len
            
            geoparser_max = int(self.max_input_chars * geoparser_ratio)
            old_max = self.max_input_chars - geoparser_max
            
            geoparser_addr = geoparser_addr[:geoparser_max]
            old_addr = old_addr[:old_max]
            
        return self.input_format.format(
            geoparser_address=geoparser_addr,
            old_address=old_addr
        )
