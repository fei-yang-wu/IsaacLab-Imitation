#!/usr/bin/env bash
# Progress of the ten capacity/batch pretrain arms.
#
# Reads each run's `metrics.jsonl`, NOT the Slurm log: the job's stdout is
# block-buffered, so a RUNNING arm's log shows nothing and a finished arm's log
# shows only its final flushed records. `metrics.jsonl` is written every log
# interval and is the only live source (found 2026-09-12, when six arms looked
# stalled for 90 minutes while their GPUs ran at 98%).
#
# Ordinary held-out next-chunk loss only. The final paired probe is a separate
# block at each arm's last record; the inactive endpoint head is never read.
# `examples` = update x batch, so the batch arms can be compared at equal
# training examples rather than equal updates.
#
#   ./report_progress.sh
set -uo pipefail
REMOTE="${REMOTE:-ice}"
ROOT="${ROOT:-/storage/ice-shared/vip-vwt/scratch-fwu91/poe_capacity_pretrain}"
LOGS="${LOGS:-/storage/ice-shared/vip-vwt/scratch-fwu91/slurm_logs}"
ARMS="${ARMS:-tail4096 tail8192 uniform2048 depth5 depth7 wide_deep long_b8192 batch16384 batch32768 batch32768_lr2}"

timeout 240 ssh "${REMOTE}" "
declare -A BATCH=( [tail4096]=8192 [tail8192]=8192 [uniform2048]=8192 [depth5]=8192 [depth7]=8192 [wide_deep]=8192 [long_b8192]=8192 [batch16384]=16384 [batch32768]=32768 [batch32768_lr2]=32768 )
for a in ${ARMS}; do
    f=\$(ls -t ${LOGS}/poe-capacity-pretrain-\${a}-s0-pretrain_*.log 2>/dev/null | head -1)
    jid=\$(basename \"\$f\" 2>/dev/null | sed -E 's/.*_([0-9]+)\.log/\1/')
    st=\$(sacct -j \"\$jid\" -X -n -o State,Elapsed 2>/dev/null | head -1 | tr -s ' ' | sed 's/^ *//')
    m=\$(find ${ROOT}/\${a}_seed0 -name metrics.jsonl 2>/dev/null | head -1)
    if [ -z \"\$m\" ]; then printf '%-16s %-24s no metrics yet\n' \"\$a\" \"\$st\"; continue; fi
    tail -1 \"\$m\" | python3 -c \"
import json,sys
d=json.load(sys.stdin); bs=\${BATCH[\$a]}
u=d.get('update') or 0; l=d.get('train/jepa_ntp_loss_eval')
print('%-16s %-24s update %-7s examples %8.1fM  ntp_loss_eval %s' % ('\$a','\$st',u,u*bs/1e6, ('%.5f'%l) if l is not None else '-'))\"
done"
