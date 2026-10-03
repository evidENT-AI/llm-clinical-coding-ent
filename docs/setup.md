# Setup

## SNOMED CT terminology

The pipeline searches SNOMED CT on a Snowstorm server and maps concepts to
ICD-10 and OPCS-4 with the NHS complex maps. The study used:

- Elasticsearch 8.11.1
- Snowstorm 10.5.1
- SNOMED CT UK Monolith Edition v42.4.0 (`uk_sct2mo_42.4.0_20260729000001Z.zip`)
- ICD-10 5th edition UK complex map (refset `999002271000000101`)
- OPCS-4.11 complex map (refset `1891651000000103`)

The Monolith release is available to UK licensees from NHS TRUD. It cannot be
redistributed, so neither the release nor the derived lookup is included here.

### Elasticsearch

```
tar xzf elasticsearch-8.11.1-*.tar.gz
cd elasticsearch-8.11.1
cat >> config/elasticsearch.yml <<'YML'
discovery.type: single-node
network.host: 127.0.0.1
xpack.security.enabled: false
xpack.ml.enabled: false
indices.query.bool.max_clause_count: 200000
YML
printf -- "-Xms16g\n-Xmx16g\n" > config/jvm.options.d/heap.options
bin/elasticsearch -d -p es.pid
```

### Snowstorm

Load the release once. The Monolith needs a higher terms limit than the default.

```
java -Xms4g -Xmx8g -jar snowstorm-10.5.1.jar --server.port=8081 \
  --elasticsearch.urls=http://localhost:9200 \
  --elasticsearch.index.max.terms.count=2000000 \
  --delete-indices --import=uk_sct2mo_42.4.0_20260729000001Z.zip --exit
```

Then serve it read-only:

```
java -Xms2g -Xmx8g -jar snowstorm-10.5.1.jar --server.port=8081 \
  --elasticsearch.urls=http://localhost:9200 --snowstorm.rest-api.readonly=true
curl 'localhost:8081/browser/MAIN/descriptions?term=tonsillitis&limit=1'
```

### ICD-10 and OPCS-4 lookup

```
python -m pipeline.classification --build --release uk_sct2mo_42.4.0_20260729000001Z.zip
python -m pipeline.classification --map 195677001 --kind diagnosis
```

The lookup is written to `data/snomed_to_icd_opcs.json`. For each concept the
code is taken from map group 1, preferring unconditional rules, excluding
codes that require an additional code, and taking the lowest priority.

## Local models

The open models were served with vLLM 0.26 on two NVIDIA RTX 5090 GPUs (32 GB
each). Llama 3.1 8B used one GPU with a 65,536-token context. Llama 4 Scout used
a 4-bit build across both GPUs with 10 GB offloaded to CPU memory; most episodes
ran with a 32,768-token context and the six longest with 49,152. The commands
are in `pipeline/serve_vllm.sh`. On these GPUs vLLM also needed:

```
export NCCL_P2P_DISABLE=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
```

## Azure OpenAI

GPT-4o and GPT-5 were separate Azure OpenAI resources inside the TRE. Set each
model's endpoint, deployment name and key in `.env` (see `.env.example`).
Without them the pipeline calls the public OpenAI API, which must only be used
with synthetic data.
