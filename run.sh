echo "Running Metropolis Hasting to get Gamma Distribution samples"
python MH_Alg/mp_sample.cpp
echo "Running Hardcore model simulation"
python Hardcore/gibbs_sample.cpp
echo "Compare with blocked gibbs sample on toy example"
python Block gibbs_sample.py

echo "Hope it was helpful!"