#/bin/bash
# run pso sunspot
python artigo_swarm_div.py -a pso -d sunspot -c 1 | tee > log/pso_sunspot_1.log &
python artigo_swarm_div.py -a pso -d sunspot -c 2 | tee > log/pso_sunspot_2.log &
python artigo_swarm_div.py -a pso -d sunspot -c 3 | tee > log/pso_sunspot_3.log &
python artigo_swarm_div.py -a pso -d sunspot -c 4 | tee > log/pso_sunspot_4.log &

# run pso airline
python artigo_swarm_div.py -a pso -d airline -c 1 | tee > log/pso_airline_1.log &
python artigo_swarm_div.py -a pso -d airline -c 2 | tee > log/pso_airline_2.log &
python artigo_swarm_div.py -a pso -d airline -c 3 | tee > log/pso_airline_3.log &
python artigo_swarm_div.py -a pso -d airline -c 4 | tee > log/pso_airline_4.log &

# run pso aws
python artigo_swarm_div.py -a pso -d aws -c 1 | tee > log/pso_aws_1.log &
python artigo_swarm_div.py -a pso -d aws -c 2 | tee > log/pso_aws_2.log &
python artigo_swarm_div.py -a pso -d aws -c 3 | tee > log/pso_aws_3.log &
python artigo_swarm_div.py -a pso -d aws -c 4 | tee > log/pso_aws_4.log &

# run pso sp500
python artigo_swarm_div.py -a pso -d sp500 -c 1 | tee > log/pso_sp500_1.log &
python artigo_swarm_div.py -a pso -d sp500 -c 2 | tee > log/pso_sp500_2.log &
python artigo_swarm_div.py -a pso -d sp500 -c 3 | tee > log/pso_sp500_3.log &
python artigo_swarm_div.py -a pso -d sp500 -c 4 | tee > log/pso_sp500_4.log &

# run pso usd
python artigo_swarm_div.py -a pso -d usd -c 1 | tee > log/pso_usd_1.log &
python artigo_swarm_div.py -a pso -d usd -c 2 | tee > log/pso_usd_2.log &
python artigo_swarm_div.py -a pso -d usd -c 3 | tee > log/pso_usd_3.log &
python artigo_swarm_div.py -a pso -d usd -c 4 | tee > log/pso_usd_4.log &

# run pso hit
python artigo_swarm_div.py -a pso -d hit -c 1 | tee > log/pso_hit_1.log &
python artigo_swarm_div.py -a pso -d hit -c 2 | tee > log/pso_hit_2.log &
python artigo_swarm_div.py -a pso -d hit -c 3 | tee > log/pso_hit_3.log &
python artigo_swarm_div.py -a pso -d hit -c 4 | tee > log/pso_hit_4.log &   