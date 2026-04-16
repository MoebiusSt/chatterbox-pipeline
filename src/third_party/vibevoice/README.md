# Embedded VibeVoice

This folder contains the embedded VibeVoice code from Microsoft.

## Why Embedded?

The original VibeVoice repository (https://github.com/microsoft/VibeVoice) has been removed from GitHub. Since VibeVoice is licensed under MIT, we have embedded the code here to ensure continued functionality of the ComfyUI wrapper.

## License

The code in this folder is licensed under the MIT License (see LICENSE file). Original copyright belongs to Microsoft Corporation.

## Modifications

The only modifications made to the original code are:
- Changed absolute imports from `vibevoice` to relative imports
- `modular/modular_vibevoice_tokenizer.py`: both ``encode`` methods of
  ``VibeVoiceAcousticTokenizerModel`` now accept ``is_final_chunk=False``
  (ignored, see below).
- Added ``VibeVoiceASRConfig`` to ``modular/configuration_vibevoice.py`` and
  ``VibeVoiceASRTextTokenizerFast`` to ``modular/modular_vibevoice_text_tokenizer.py``
  to enable local (non ``trust_remote_code``) loading of VibeVoice-ASR.

### Why the ``is_final_chunk`` shim?

Upstream ``modeling_vibevoice_asr.py`` (released after this vendor snapshot)
passes ``is_final_chunk=True`` into ``tokenizer.encode(...)`` to signal the
final streaming chunk. Our single-pass ASR wrapper does not use streaming, so
the vendored tokenizer silently ignores the kwarg (``del is_final_chunk``)
instead of raising ``TypeError``. This keeps the vendored modeling file
untouched while remaining compatible with the current processor/model pair.

If you ever refresh these files from upstream, keep the
``is_final_chunk=False`` parameter on both ``encode`` signatures or the
VibeVoice-ASR path will regress with
``unexpected keyword argument 'is_final_chunk'``.

## Note

This is a preservation copy to ensure the continued availability of VibeVoice for the ComfyUI community.
