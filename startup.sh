#!/bin/sh

# # # # Iniciar el servicio de Ollama en segundo plano
# # # ollama serve &
# # ollama serve
# # # echo "Mistral Descargando.................................................................."
# # # ollama pull mistral

# # # # Esperar unos segundos para asegurar que Ollama esté en funcionamiento
# # # sleep 5

# # # # Ejecutar el modelo llama3
# # #ollama run mistral

# #####################################3
# #!/bin/sh

# # Iniciar el servicio Ollama en segundo plano
# echo "Arrancando ollama server..."
# ollama serve

# # # Esperar a que Ollama esté listo
# # echo "Esperando a que Ollama esté listo..."
# # #until ollama models >/dev/null 2>&1; do
# # until ollama models; do
# #   sleep 2
# # done

# # Descargar el modelo mistral
# echo "Descargando el modelo mistral..."
# ollama pull mistral

# # Mantener el servicio activo (necesario para que el contenedor no se detenga)
# wait


##########################
# Verificar que Ollama responde correctamente en el endpoint /api/ps
echo "Verificando que Ollama esté funcionando..."
until curl -s http://ollama:11434/api/ps; do
  echo "Ollama no está listo aún. Esperando..."
  sleep 2
done

echo "Ollama está funcionando."

# Ejecutar el pull del modelo Mistral
echo "Descargando el modelo Mistral..."
curl -X POST http://ollama:11434/api/pull -d '{
  "name": "mistral"
}'