from src.models.base import Model
from src.models.llm_models import HuggingFaceModel, OpenaiModel, MistralModel, HFOpenaiModel

def get_model(model_name: str,
              apply_defense_methods: bool,
              auth_token: str, 
              device: str, 
              system_prompt: str,
              temperature: float,
              top_p: float) -> Model:
    """
    Factory method to get the appropriate model based on the given parameters.

    Args:
        model_name (str): The name of the model.
        auth_token (str): The authentication token.
        device (str): The device to run the model on.
        system_prompt (str): The system prompt to use.
        temperature (float): The temperature for generating text.
        top_p (float): The top-p value for generating text.

    Returns:
        Model: An instance of the appropriate model based on the given parameters.

    Raises:
        ValueError: If the model_name is unknown.
    """
    model_name = model_name.lower()
    if model_name in HuggingFaceModel.model_details:
        return HuggingFaceModel(
            auth_token=auth_token,
            device=device, 
            system_prompt=system_prompt, 
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods)
    elif model_name in OpenaiModel.model_details:
        model =  OpenaiModel(
            auth_token=auth_token, 
            device=device, 
            system_prompt=system_prompt, 
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods
        )
        return model
    elif model_name in MistralModel.model_details:
        return MistralModel(
            auth_token=auth_token,
            device=device,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods
        )
    elif model_name in HFOpenaiModel.model_details:
        return HFOpenaiModel(
            auth_token=auth_token,
            device=device,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            apply_defense_methods=apply_defense_methods
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
