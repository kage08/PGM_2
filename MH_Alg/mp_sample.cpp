#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>

std::vector<double > mp_sample(double k=1.0,double theta=2.0,double sigma=1.0,int T=100000){
    std::vector<double > samples;
    double curr_state = 1;
    double next_state;
    double acceptance_probab;
    samples.push_back(curr_state);
    double gap;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,sigma);
    std::uniform_real_distribution<double> u_distribution(0.0,1.0);

    for(int i=0; i<T; i++){
        gap = distribution(generator);
        next_state = curr_state + gap;
        if(next_state<=0.0){
            samples.push_back(curr_state);
            continue;
        }
        acceptance_probab = pow(next_state/curr_state,k-1)*exp((curr_state-next_state)/theta);
        curr_state = (u_distribution(generator)<acceptance_probab)?next_state : curr_state;
        samples.push_back(curr_state);
    }

    return samples;
}




int main(int argc, char* argv[]){
    if(argc<6){
        std::cout<<"Usage: ./mp_sample <k> <theta> <sigma> <no_samples> <save_flie>\n";
        exit(1);
    }
    int k = atof(argv[1]);
    int theta = atof(argv[2]);
    int sigma = atof(argv[3]);
    int T = atoi(argv[4]);
    std::vector<double > samples = mp_sample(k, theta, sigma, T);
    std::cout<<"Done"<<"\n";

    std::ofstream wfile;
    wfile.open(argv[5]);
    for(int i=0; i<samples.size(); i++){
        wfile << samples[i]<<"\n";
    }
    wfile.close();
    return 0;
}