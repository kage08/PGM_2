#include <iostream>
#include <cstdlib>
#include <vector>
#include <fstream>

std::vector< std::vector<bool> > gibbs_sample(int grid_len, int T=1000000, bool r=false){
    std::vector< std::vector<bool> > states;
    std::vector<bool> curr_state(grid_len*grid_len, false);
    bool is_valid;
    int toss;

    for(int k=0; k<T; k++){
        is_valid = true;
        int i,j,t = 0;
        while(t<grid_len*grid_len){
            if(!r){
                i = t/grid_len;
                j = t%grid_len;
            }
            else{
                i = rand() % grid_len;
                j = rand() % grid_len;
            }
        
            if (curr_state[i*grid_len+j]==false){
                if(i-1>= 0 && curr_state[(i-1)*grid_len+j]==true) is_valid=false;
                if(j-1>= 0 && curr_state[i*grid_len+j-1]==true) is_valid=false;
                if(i+1< grid_len && curr_state[(i+1)*grid_len+j]==true) is_valid=false;
                if(j+1< grid_len && curr_state[i*grid_len+j+1]==true) is_valid=false;
            }
            if(is_valid){
                toss = rand() % 2;
                curr_state[i*grid_len+j] = (bool)toss;
            }
            t++;
        }
        std::vector <bool> copy_state(curr_state);
        states.push_back(copy_state);
    }


    return states;
}

int main(int argc, char* argv[]){
    if(argc<4){
        std::cout<<"Usage: ./gibbs_sample <grid_len> <no_samples> <save_flie>\n";
        exit(1);
    }
    int grid_len = atoi(argv[1]);
    int T = atoi(argv[2]);
    std::vector< std::vector<bool> > samples = gibbs_sample(grid_len, T);

    std::ofstream wfile;
    wfile.open(argv[3]);
    for(int i=0; i<samples.size(); i++){
        for(int j=0; j<samples[i].size(); j++)
            wfile << samples[i][j]<<" ";
        wfile<<"\n";
    }
    wfile.close();
    return 0;
}