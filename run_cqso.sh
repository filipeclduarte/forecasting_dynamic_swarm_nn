#/bin/bash
# run cqso sunspot
python artigo_swarm_div.py -a cqso -d sunspot -c 1 | tee > log/cqso_sunspot_1.log &
python artigo_swarm_div.py -a cqso -d sunspot -c 2 | tee > log/cqso_sunspot_2.log &
python artigo_swarm_div.py -a cqso -d sunspot -c 3 | tee > log/cqso_sunspot_3.log &
python artigo_swarm_div.py -a cqso -d sunspot -c 4 | tee > log/cqso_sunspot_4.log &

# run cqso airline
python artigo_swarm_div.py -a cqso -d airline -c 1 | tee > log/cqso_airline_1.log &
python artigo_swarm_div.py -a cqso -d airline -c 2 | tee > log/cqso_airline_2.log &
python artigo_swarm_div.py -a cqso -d airline -c 3 | tee > log/cqso_airline_3.log &
python artigo_swarm_div.py -a cqso -d airline -c 4 | tee > log/cqso_airline_4.log &

# run cqso aws
python artigo_swarm_div.py -a cqso -d aws -c 1 | tee > log/cqso_aws_1.log &
python artigo_swarm_div.py -a cqso -d aws -c 2 | tee > log/cqso_aws_2.log &
python artigo_swarm_div.py -a cqso -d aws -c 3 | tee > log/cqso_aws_3.log &
python artigo_swarm_div.py -a cqso -d aws -c 4 | tee > log/cqso_aws_4.log &

# run cqso sp500
python artigo_swarm_div.py -a cqso -d sp500 -c 1 | tee > log/cqso_sp500_1.log &
python artigo_swarm_div.py -a cqso -d sp500 -c 2 | tee > log/cqso_sp500_2.log &
python artigo_swarm_div.py -a cqso -d sp500 -c 3 | tee > log/cqso_sp500_3.log &
python artigo_swarm_div.py -a cqso -d sp500 -c 4 | tee > log/cqso_sp500_4.log &

# run cqso usd
python artigo_swarm_div.py -a cqso -d usd -c 1 | tee > log/cqso_usd_1.log &
python artigo_swarm_div.py -a cqso -d usd -c 2 | tee > log/cqso_usd_2.log &
python artigo_swarm_div.py -a cqso -d usd -c 3 | tee > log/cqso_usd_3.log &
python artigo_swarm_div.py -a cqso -d usd -c 4 | tee > log/cqso_usd_4.log &

# run cqso hit
python artigo_swarm_div.py -a cqso -d hit -c 1 | tee > log/cqso_hit_1.log &
python artigo_swarm_div.py -a cqso -d hit -c 2 | tee > log/cqso_hit_2.log &
python artigo_swarm_div.py -a cqso -d hit -c 3 | tee > log/cqso_hit_3.log &
python artigo_swarm_div.py -a cqso -d hit -c 4 | tee > log/cqso_hit_4.log &