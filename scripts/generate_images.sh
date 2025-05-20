#!/bin/bash

# Script para gerar imagens de exemplo para o SISTEMA_EMG
# Este script cria imagens simples para ilustrar a documentação

# Configurações
OUTPUT_DIR="docs/images"
WIDTH=800
HEIGHT=400

# Verifica se o diretório de saída existe
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Função para gerar uma imagem simples com texto
generate_image() {
    local filename="$1"
    local title="$2"
    local color="$3"
    
    convert -size ${WIDTH}x${HEIGHT} xc:"$color" \
        -gravity center \
        -pointsize 40 \
        -font "DejaVu-Sans-Bold" \
        -fill white \
        -annotate 0 "$title" \
        "$OUTPUT_DIR/$filename"
    
    echo "Imagem gerada: $OUTPUT_DIR/$filename"
}

# Gera imagens para os diferentes modos
generate_image "banner.png" "SISTEMA_EMG" "navy"
generate_image "simulation_mode.png" "Modo de Simulação" "darkgreen"
generate_image "collection_mode.png" "Modo de Coleta" "darkred"
generate_image "training_mode.png" "Modo de Treinamento" "darkorange"
generate_image "execution_mode.png" "Modo de Execução" "darkblue"

echo "Todas as imagens foram geradas com sucesso!"
