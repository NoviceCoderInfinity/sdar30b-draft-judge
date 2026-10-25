#!/usr/bin/env bash
# swap in the new orchestrator right after the science -> code handoff (before the code engine decodes anything)
cd /workspace/dmge
until tail -3 logs/c1b/waveA.log | grep -q "gen code until"; do sleep 5; done
sleep 2
chmod +x code/corpus1b/run_wave_a.sh.new && mv code/corpus1b/run_wave_a.sh.new code/corpus1b/run_wave_a.sh
tmux kill-session -t wa; sleep 5
tmux new -d -s wa 'bash code/corpus1b/run_wave_a.sh 2>&1 | tee -a logs/c1b/waveA.log'
echo "swapped at $(date -u +%H:%M:%S) UTC" >> logs/c1b/dl/swap.log
