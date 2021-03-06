//usr/bin/g++ compute_values.cpp -std=c++1z -fopenmp -march=native -O3 && ./a.out;
#include <algorithm>
#include <utility>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <map>
#include <fstream>
#include <stdexcept>
#include "csv.hpp"
#include<eigen3/Eigen/Dense>

using namespace std;

/// state type: (positive, negetive)
typedef pair<const int, const int> state_t;


template<class T> std::ostream & operator<<(std::ostream &os, const std::vector<T>& vec){
    for(const auto& p : vec)
        os << p << " ";
    return os;
}

/// Print a list
template<class T1, class T2> std::ostream & operator<<(std::ostream &os, const std::pair<T1,T2>& p){
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}

/// \returns two possible outcome states and probabilities
pair<pair<state_t,double>, pair<state_t,double>> transition(state_t state){
    const double positiveprob = double(state.first) / double(state.first + state.second);
    const double negativeprob = 1.0 - positiveprob;
    const state_t positive_state = make_pair(state.first + 1, state.second);
    const state_t negative_state = make_pair(state.first, state.second + 1);
    return make_pair(make_pair(positive_state, positiveprob), make_pair(negative_state, negativeprob));
}

/// \returns number of steps to the horizon
/// horizon = is one state
int steps_to_end(uint horizon, state_t state){
    return 2 + horizon - (state.first + state.second);
}

/// \param timestep 0-based time step for the value function
/// \returns UCB estimate for a value function - expected mean return
double ucb_benefit(state_t state, long timestep){
    assert(timestep >= 0);
    const double alpha = 2.0;
    return sqrt(alpha * log(timestep) / (2.0*double(state.first + state.second)));
}

/// \returns gittins index value - expected return
double gittins_benefit(const map<state_t,double>& state2gittins, state_t state){
    double expected_reward = double(state.first) / double(state.first + state.second);
    return state2gittins.at(state) - expected_reward;
}

/// Type of the index used for the computation
enum class IndexType{ UCB, Gittins };

/// Determine the name for the output file name
map<state_t, double> load_gittins(const string& filename){
    io::CSVReader<3> gittinsfile(filename);

    gittinsfile.read_header(io::ignore_no_column, "Positive", "Negative", "Index");

    long positive, negative;
    double index;
    map<state_t, double> state2gittins;

    while(gittinsfile.read_row(positive,negative,index)){
        state2gittins[std::make_pair(positive,negative)] = index;
    }
    return state2gittins;
}

int main(){
    // -- initialize ---------------------------------------------------
    // number of steps (horizon = 1 is 1 state)
    const uint horizon = 402;
    const IndexType index = IndexType::UCB;

    // output file name
    string output_filename;
    const string input_gittins_filename = "gittins.csv";
    map<state_t, double> state2gittins; // state to Gittins index

    // update parameters based on the index that is being used
    if(index == IndexType::UCB){
        cout << "Using UCB index ..." << endl;
        output_filename = "ucb_value.csv";
    }
    else if(index == IndexType::Gittins){
        cout << "Using Gittins index ..." << endl;
        output_filename =  "gittins_value.csv";
        cout << "Loading Gittins indexes ..." << endl;
        state2gittins = load_gittins(input_gittins_filename);
    }
    else{
        throw out_of_range("Unknown index type");
    }

    cout << "Horizon: " << horizon << endl;
    cout << "Output file: \"" << output_filename << "\"" << endl;

    auto start = chrono::steady_clock::now();
    // -- generate states -------------------------------------------------
    std::vector<state_t> states;
    cout << "Generating states ..." << endl;
    for(int level=0; level < horizon; level++){
        for(int poscount=1; poscount < level+2; poscount++){
            const int negcount = level+2 - poscount;
            states.push_back(std::make_pair(poscount, negcount));
        }
    }
    const size_t state_count = states.size();
    cout << "Number of states: " << state_count << endl;

    // state map
    std::cout << "Constructing state map ..." << endl;
    map<state_t, int> state2index;
    for(size_t i = 0; i < states.size(); ++i)
        state2index[states[i]] = i;

    // value function (assumes that it is initialized to all 0s)
    Eigen::ArrayXXd valuefunction = Eigen::ArrayXXd::Zero(horizon, state_count); //timestep, state
    // the value function satisfies:
    // v_{t+1}(s) = l_t(s,a) - B(s,a)
    // where B(s,a) = UCB(s,a) - r(s,a)
    // and l_t = expected next value (without the immediate reward)

    // -- compute state values -------------------------------------------------
    cout << "Computing state values ... " << endl;
    // just assume that the value function in the last step is 0
    for(long t = horizon-1; t >= 0; t--){

        // this is the offset for the current horizon
        double offset = 0;

        // compute state value functions
        for(long istate = state_count-1; istate >= 0; istate--){
            auto state = states[istate];
            // an impossible state!
            if(state.first + state.second - 2 > t){
                valuefunction(t,istate) = nan("");
                continue;
            }

            // next states
            auto nextstates = transition(state);
            pair<state_t, double>
                positive_sp = nextstates.first,
                negative_sp = nextstates.second;

            // an edge state - initialize it to 0
            if(state.first + state.second - 2 == t){
                valuefunction(t,istate) = 0;
            }
            // compute the value otherwise (no continue to be able to compute the offset)
            else{
                // value of taking the uncertain function
                // states beyond the horizon have value 0
                double lvaluet =
                    positive_sp.second * valuefunction(t, state2index[positive_sp.first]) +
                    negative_sp.second * valuefunction(t, state2index[negative_sp.first]);

                double benefit;
                if(index == IndexType::UCB) benefit = ucb_benefit(state, t);
                else if(index == IndexType::Gittins) benefit = gittins_benefit(state2gittins, state);
                else throw out_of_range("Unknown index type");

                valuefunction(t,istate) =  lvaluet - benefit;
            }

            // compute an appropriate offset so that the value function is an over-estimate
            // this is a simplistic version and it will result in a large over-estimate for the
            // value function with mutliple arms. It may be better to compute the offsets
            // using approximate dynamic programming
            // v_{t}(s) >= q_{t}(s,a)
            if(t < horizon-1){ // ignore the last step
                double qvalue =
                    positive_sp.second * (1 + valuefunction(t+1, state2index[positive_sp.first])) +
                    negative_sp.second * (0 + valuefunction(t+1, state2index[negative_sp.first]));

                offset = max(offset, qvalue - valuefunction(t,istate));
            }
        }

        // update the vakue function according to offsets (if non-zero)
        if(offset > numeric_limits<double>::epsilon()){
            valuefunction.row(t) += offset;
        }
        cout << t << ":  " << offset << endl;
    }

    auto end = chrono::steady_clock::now();
    auto diff = end - start;

    // -- process output  -------------------------------------------------
    cout << "Duration (computations): " << chrono::duration <double, milli> (diff).count() << " ms" << endl;
    cout << "Writing results to a csv: " << output_filename << endl;
    ofstream fout(output_filename);
    fout << "Time,Positive,Negative,Value" << endl;
    for (long t = 0; t < horizon; t++){
        for (size_t istate = 0; istate < state_count; istate++) {
            auto state = states[istate];
            if(!isnan(valuefunction(t,istate)))
                fout << t << "," << state.first << "," << state.second << "," << valuefunction(t,istate) << endl;
        }
    }

    return 0;

}
