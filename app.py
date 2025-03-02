import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO

# Configuração da GPU (se disponível)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregamento do modelo (cacheado para evitar downloads repetidos)
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

st.title("Gerador de Imagens com Stable Diffusion")

prompt = st.text_input("Digite seu prompt:")

if st.button("Gerar Imagem"):
    if prompt:
        with st.spinner("Gerando imagem..."):
            try:
                image = pipe(prompt).images[0]
                st.image(image, caption="Imagem Gerada", use_column_width=True)

                # Opção de download
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                st.download_button(
                    label="Baixar Imagem",
                    data=buffered.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Erro ao gerar imagem: {e}")
    else:
        st.warning("Por favor, digite um prompt.")

# Instruções para Streamlit Cloud
st.markdown("""
---
**Instruções para Streamlit Cloud:**

1.  **Requisitos:**
    * Certifique-se de ter `streamlit`, `diffusers`, `transformers`, `accelerate`, `torch` e `safetensors` no seu arquivo `requirements.txt`.
2.  **GPU:**
    * Para usar a GPU no Streamlit Cloud, selecione a instância "GPU Basic" ou superior durante a criação do aplicativo. (Recomendado para melhor desempenho).
3.  **Memória:**
    * Modelos de difusão são grandes. Se você enfrentar problemas de memória, considere usar um modelo menor ou uma instância de Streamlit Cloud com mais memória.
4. **Cache:** o `@st.cache_resource` foi adicionado para melhorar a performance.
5. **Download:** Adicionado um botão para fazer o download da imagem gerada.
6. **safetensors:** O argumento `use_safetensors=True` foi adicionado para acelerar o carregamento do modelo e utilizar um formato mais seguro para armazenar os pesos do modelo.
""")
