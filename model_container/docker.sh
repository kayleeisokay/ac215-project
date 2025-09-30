# build image
docker build --no-cache -t baseline-model -f Dockerfile .

# run container
docker run \
    --rm \
    --name baseline-model-container \
    -ti \
    -v ~/secrets/dosewise-473716-9f4874e812d6.json:/app/dosewise-473716-9f4874e812d6.json \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/dosewise-473716-9f4874e812d6.json \
    baseline-model