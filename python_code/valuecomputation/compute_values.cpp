//usr/bin/g++ compute_values.cpp -fopenmp -march=native -O3 && ./a.out;
#include<algorithm>
#include<utility>
#include<iostream>
#include<chrono>
#include<vector>
#include<cmath>
#include<map>
#include<fstream>
#include<eigen3/Eigen/Dense>

using namespace std;

/// state type: (positive, negetive)
typedef pair<const int, const int> state_t;

template<class T> std::ostream & operator<<(std::ostream &os, const std::vector<T>& vec){
    for(const auto& p : vec)
        cout << p << " ";
    return os;
}

template<class T1, class T2> std::ostream & operator<<(std::ostream &os, const std::pair<T1,T2>& p){
    cout << "(" << p.first << ", " << p.second << ")";
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
    return sqrt(alpha * log(timestep+1) / (2.0*double(state.first + state.second)));
}

int main(){
    // -- initialize ---------------------------------------------------
    // number of steps (horizon = 1 is 1 state)
    const uint horizon = 20;
    // output file name
    const string output_filename = "ucb_value.csv";

    cout << "Horizon: " << horizon << endl;

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
    Eigen::MatrixXd valuefunction(horizon, state_count); //timestep, state
    // the value function satisfies:
    // v_t(s) = B(s,a) - l_t(s,a) + v_{t+1}(s)
    // where B(s,a) = UCB(s,a) - r(s,a)
    // and l_t =

    // -- compute state values -------------------------------------------------
    cout << "Computing state values ... " << endl;
    // just assume that the value function in the last step is 0
    for(long t = horizon-2; t >= 0; t--){
        // compute state value functions
        #pragma omp parallel for
        for(long istate=state_count-1; istate >= 0; istate--){
            auto state = states[istate];
            if(state.first + state.second - 2 > t){
                valuefunction(t,istate) = nan("");
                continue;
            }

            // next states
            auto nextstates = transition(state);
            pair<state_t, double>
                positive_sp = nextstates.first,
                negative_sp = nextstates.second;

            // value of taking the uncertain function
            // states beyond the horizon have value 0
            auto lvalue =
                positive_sp.second * valuefunction(t+1, state2index[positive_sp.first]) +
                negative_sp.second * valuefunction(t+1, state2index[negative_sp.first]);

            if(t == 0)
                cout << "ucb benefit:" << ucb_benefit(state, t) << endl;
            valuefunction(t,istate) = ucb_benefit(state, t) - lvalue + valuefunction(t+1,istate);
        }
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
