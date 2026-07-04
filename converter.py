import torch
from onnxruntime.quantization import quantize_dynamic, QuantType

# IMPORTANTE: Importe as classes dos seus modelos aqui!
# Exemplo: from seus_arquivos import SRCNN, EDSR, ESRGAN
from models import SRCNN, EDSRBaseline, ESRGANGenerator 

def exportar_e_quantizar(modelo_pytorch, nome_base, caminho_pesos, tamanho_entrada=(1, 3, 256, 256)):
    print(f"\n--- Processando {nome_base} ---")
    
    # 1. Carrega o modelo PyTorch
    modelo = modelo_pytorch()
    modelo.load_state_dict(torch.load(caminho_pesos, map_location=torch.device('cpu')))
    modelo.eval()
    
    # 2. Cria uma entrada falsa para rastrear a rede
    dummy_input = torch.randn(tamanho_entrada)
    
    arquivo_onnx = f"{nome_base}.onnx"
    arquivo_quantizado = f"{nome_base}_quantizado.onnx"
    
    # 3. Exporta para ONNX original
    print("1/2 Exportando para ONNX...")
    torch.onnx.export(
        modelo, 
        dummy_input, 
        arquivo_onnx, 
        input_names=["input"], 
        output_names=["output"],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    
    # 4. Quantiza para 8-bits (reduz o tamanho e aumenta a velocidade na web)
    print("2/2 Quantizando para 8-bits...")
    quantize_dynamic(
        model_input=arquivo_onnx,
        model_output=arquivo_quantizado,
        weight_type=QuantType.QUInt8
    )
    print(f"Pronto! Arquivo final gerado: {arquivo_quantizado}")

# --- EXECUÇÃO ---
# Substitua pelas suas classes e caminhos reais dos pesos .pth
if __name__ == "__main__":
    #exportar_e_quantizar(SRCNN, "srcnn", "pesos/srcnn.pth")
    #exportar_e_quantizar(EDSRBaseline, "edsr", "pesos/edsr.pth")
    exportar_e_quantizar(ESRGANGenerator, "esrgan", "resultados_esrgan/esrgan_final_interpolated.pth")
    print("Descomente as linhas acima e configure seus modelos para rodar!")