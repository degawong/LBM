from pathlib import Path
import hashlib

import torch
from transformers import CLIPTokenizer, CLIPTextModel


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

PROMPT = "brighten the person and darken the background"

CACHE_DIR = Path("prompt_cache")

DTYPE = torch.bfloat16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Prompt Embedding Cache
# ============================================================

class PromptEmbeddingCache:
    def __init__(
        self,
        tokenizer,
        text_encoder,
        device,
        cache_dir="prompt_cache",
        dtype=torch.bfloat16,
    ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.memory_cache = {}

    def _cache_path(self, prompt):
        key = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()[:16]

        return self.cache_dir / f"{key}.pt"

    @torch.no_grad()
    def get(self, prompt):
        # ----------------------------------------------------
        # 1. Memory cache
        # ----------------------------------------------------
        if prompt in self.memory_cache:
            print(f"[Memory Cache] {prompt}")
            return self.memory_cache[prompt]

        # ----------------------------------------------------
        # 2. Disk cache
        # ----------------------------------------------------
        cache_path = self._cache_path(prompt)

        if cache_path.exists():
            print(f"[Disk Cache] {cache_path}")

            encoder_hidden_states = torch.load(
                cache_path,
                map_location=self.device,
                weights_only=True,
            )

            encoder_hidden_states = encoder_hidden_states.to(
                device=self.device,
                dtype=self.dtype,
            )

            self.memory_cache[prompt] = encoder_hidden_states

            return encoder_hidden_states

        # ----------------------------------------------------
        # 3. First time: encode prompt
        # ----------------------------------------------------
        print(f"[Encoding] {prompt}")

        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)

        encoder_hidden_states = self.text_encoder(
            input_ids
        )[0]

        # ----------------------------------------------------
        # SD1.5 CLIP output:
        #
        # [1, 77, 768]
        # ----------------------------------------------------
        encoder_hidden_states = encoder_hidden_states.to(
            device=self.device,
            dtype=self.dtype,
        ).detach()

        print(
            f"Embedding shape: "
            f"{tuple(encoder_hidden_states.shape)}"
        )

        print(
            f"Embedding dtype: "
            f"{encoder_hidden_states.dtype}"
        )

        # ----------------------------------------------------
        # 4. Save CPU tensor to disk
        # ----------------------------------------------------
        torch.save(
            encoder_hidden_states.cpu(),
            cache_path,
        )

        print(
            f"[Saved] {cache_path}"
        )

        # ----------------------------------------------------
        # 5. Memory cache
        # ----------------------------------------------------
        self.memory_cache[prompt] = encoder_hidden_states

        return encoder_hidden_states


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("SD1.5 Prompt Embedding Generator")
    print("=" * 60)

    print(f"Model : {MODEL_ID}")
    print(f"Prompt: {PROMPT}")
    print(f"Device: {DEVICE}")
    print(f"Dtype : {DTYPE}")

    # --------------------------------------------------------
    # Load SD1.5 tokenizer
    # --------------------------------------------------------

    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_ID,
        subfolder="tokenizer",
    )

    # --------------------------------------------------------
    # Load SD1.5 CLIP Text Encoder
    # --------------------------------------------------------

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
    )

    text_encoder = text_encoder.to(DEVICE)

    text_encoder.eval()

    # --------------------------------------------------------
    # Freeze Text Encoder
    # --------------------------------------------------------

    for param in text_encoder.parameters():
        param.requires_grad_(False)

    # --------------------------------------------------------
    # Create cache
    # --------------------------------------------------------

    cache = PromptEmbeddingCache(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=DEVICE,
        cache_dir=CACHE_DIR,
        dtype=DTYPE,
    )

    # --------------------------------------------------------
    # Generate / Load embedding
    # --------------------------------------------------------

    embedding = cache.get(PROMPT)

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print(f"Prompt : {PROMPT}")
    print(f"Shape  : {tuple(embedding.shape)}")
    print(f"Dtype  : {embedding.dtype}")
    print(f"Device : {embedding.device}")

    cache_path = cache._cache_path(PROMPT)

    print(f"File   : {cache_path}")

    # --------------------------------------------------------
    # Verify saved file
    # --------------------------------------------------------

    saved_embedding = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )

    print()
    print("Saved embedding:")
    print(f"Shape  : {tuple(saved_embedding.shape)}")
    print(f"Dtype  : {saved_embedding.dtype}")
    print(f"Device : {saved_embedding.device}")


if __name__ == "__main__":
    main()