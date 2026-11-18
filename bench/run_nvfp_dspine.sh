#!/bin/bash
# NVFP 生产制品 + DSpine 移植 bench:
#   - nvfp-dspine-d7/d15 : NVFP 目标 + DSpine drafter (合并制品)
#   - nvfp-mtp-d4        : 同 NVFP 制品, MTP 4 (生产路径同参数), 作同制品对照
# 全部 --max-context 185000 --kv-dtype int8 --lm-head-draft, V100 全功耗 250W。
# 注: serve 路径非 KV 开销 (~3.55 GiB) + 1 GiB headroom 下, 21.2 GiB 权重使 200k int8 KV
# (6.29 GiB) 超限 (缺 0.38 GiB), 195k (6.47 GiB, 含 draft 窗口) 也缺 0.09 GiB, 取 185k
# (KV ~5.94 GiB, 总 ~9.5 GiB, 余量 1.4 GiB)。最长 bench prompt 193K 超出 185k, 由引擎
# 截断; 其余 7.3K/31K/93K 三个点在 185k 内。DSpine 纯文本, 无 --vision。
set -u
cd /home/max/code/docker/ai/ninfer-v100
OUT=/tmp/nov18b-nvfp-dspine
mkdir -p "$OUT"
BENCH=$PWD/bench
LOG="$OUT/bench.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

C=ninfer-nvfp-dspine-bench
SERVE=/src/ninfer-serve-bin
MODEL=/models/Qwen3.8-27B-Coder390-NVFP-dspine.ninfer
ENVD='-e LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu'

wait_health(){ local d=$((SECONDS+1500)); while [ $SECONDS -lt $d ]; do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }

setup_container(){
  docker rm -f $C >/dev/null 2>&1
  docker run -d --name $C --gpus '"device=1"' -p 8000:8000 \
    -v /home/max/code/ninfer-v100:/src:ro \
    -v /mnt/data/dockerdata/llama-cpp/models:/models:ro \
    $ENVD nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 sleep infinity >>"$LOG" 2>&1
  docker exec $C bash -c 'apt-get update -q >/dev/null 2>&1; apt-get install -y -q libavformat60 libavcodec60 libavutil58 libswscale7 libcurl4 >/dev/null 2>&1'
  log "container ready"
}

start(){ # tag spec draft-tokens
  local tag=$1 spec=$2 dt=$3
  docker exec $C pkill -f ninfer-serve-bin >/dev/null 2>&1; sleep 3
  log "start $tag (spec $spec, draft-tokens $dt)"
  docker exec -d $C bash -c "$SERVE $MODEL --host 0.0.0.0 --port 8000 --api-key sk-itzy --model-id qwen \
    --max-context 185000 --kv-dtype int8 --max-concurrency 2 --prefill-chunk 2048 \
    --lm-head-draft --spec $spec --draft-tokens $dt > /tmp/serve.log 2>&1"
  if wait_health; then log "$tag ready after $((SECONDS-3))s"; else
    log "$tag HEALTH TIMEOUT"; docker exec $C tail -30 /tmp/serve.log | tee -a "$LOG"; return 1; fi
  sleep 5
}

bench(){ # tag
  local tag=$1
  python3 "$BENCH/v100/decode_bench.py" --url http://127.0.0.1:8000 --model qwen --key sk-itzy \
    --kind code --blocks 29 --seeds 1,2 --max-tokens 512 --tag "$tag" --out "$OUT/rows.jsonl" >>"$LOG" 2>&1 \
    || log "$tag code29 FAILED"
  python3 "$BENCH/v100/decode_bench.py" --url http://127.0.0.1:8000 --model qwen --key sk-itzy \
    --kind code --blocks 124 --seeds 1 --max-tokens 512 --tag "$tag" --out "$OUT/rows.jsonl" >>"$LOG" 2>&1 \
    || log "$tag code124 FAILED"
  python3 "$BENCH/v100/decode_bench.py" --url http://127.0.0.1:8000 --model qwen --key sk-itzy \
    --kind code --blocks 473 --seeds 1 --max-tokens 512 --tag "$tag" --out "$OUT/rows.jsonl" >>"$LOG" 2>&1 \
    || log "$tag code473 FAILED"
  python3 "$BENCH/v100/decode_bench.py" --url http://127.0.0.1:8000 --model qwen --key sk-itzy \
    --kind zh-doc --blocks 1162 --seeds 1 --max-tokens 256 --tag "$tag" --out "$OUT/rows.jsonl" >>"$LOG" 2>&1 \
    || log "$tag zhdoc1162 FAILED"
}

log "=== NVFP+DSpine bench start (V100 全功耗 250W, ctx 185k) ==="
setup_container
if start nvfp-dspine-d7  dspine 7;  then bench nvfp-dspine-d7;  fi
if start nvfp-dspine-d15 dspine 15; then bench nvfp-dspine-d15; fi
if start nvfp-mtp-d4     mtp    4;  then bench nvfp-mtp-d4;     fi
docker rm -f $C >/dev/null 2>&1
log "=== done ==="
