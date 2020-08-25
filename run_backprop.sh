#/bin/bash
# run backprop sunspot
python artigo_swarm_div.py -a backprop -d sunspot -c 1 | tee > log/backprop_sunspot_1.log &
python artigo_swarm_div.py -a backprop -d sunspot -c 2 | tee > log/backprop_sunspot_2.log &
python artigo_swarm_div.py -a backprop -d sunspot -c 3 | tee > log/backprop_sunspot_3.log &
python artigo_swarm_div.py -a backprop -d sunspot -c 4 | tee > log/backprop_sunspot_4.log &

# run backprop airline
python artigo_swarm_div.py -a backprop -d airline -c 1 | tee > log/backprop_airline_1.log &
python artigo_swarm_div.py -a backprop -d airline -c 2 | tee > log/backprop_airline_2.log &
python artigo_swarm_div.py -a backprop -d airline -c 3 | tee > log/backprop_airline_3.log &
python artigo_swarm_div.py -a backprop -d airline -c 4 | tee > log/backprop_airline_4.log &

# run backprop aws
python artigo_swarm_div.py -a backprop -d aws -c 1 | tee > log/backprop_aws_1.log &
python artigo_swarm_div.py -a backprop -d aws -c 2 | tee > log/backprop_aws_2.log &
python artigo_swarm_div.py -a backprop -d aws -c 3 | tee > log/backprop_aws_3.log &
python artigo_swarm_div.py -a backprop -d aws -c 4 | tee > log/backprop_aws_4.log &

# run backprop sp500
python artigo_swarm_div.py -a backprop -d sp500 -c 1 | tee > log/backprop_sp500_1.log &
python artigo_swarm_div.py -a backprop -d sp500 -c 2 | tee > log/backprop_sp500_2.log &
python artigo_swarm_div.py -a backprop -d sp500 -c 3 | tee > log/backprop_sp500_3.log &
python artigo_swarm_div.py -a backprop -d sp500 -c 4 | tee > log/backprop_sp500_4.log &

# run backprop usd
python artigo_swarm_div.py -a backprop -d usd -c 1 | tee > log/backprop_usd_1.log &
python artigo_swarm_div.py -a backprop -d usd -c 2 | tee > log/backprop_usd_2.log &
python artigo_swarm_div.py -a backprop -d usd -c 3 | tee > log/backprop_usd_3.log &
python artigo_swarm_div.py -a backprop -d usd -c 4 | tee > log/backprop_usd_4.log &

# run backprop hit
python artigo_swarm_div.py -a backprop -d hit -c 1 | tee > log/backprop_hit_1.log &
python artigo_swarm_div.py -a backprop -d hit -c 2 | tee > log/backprop_hit_2.log &
python artigo_swarm_div.py -a backprop -d hit -c 3 | tee > log/backprop_hit_3.log &
python artigo_swarm_div.py -a backprop -d hit -c 4 | tee > log/backprop_hit_4.log &