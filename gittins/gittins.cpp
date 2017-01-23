//usr/bin/g++ gittins.cpp -fopenmp -march=native -O3 && ./a.out;
#include<iostream>
#include<vector>
#include<utility>
#include<cmath>
#include<map>
#include<chrono>
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
inline pair<pair<state_t,double>, pair<state_t,double>> transition(state_t state){
    const double positiveprob = double(state.first) / double(state.first + state.second);
    const double negativeprob = 1.0 - positiveprob;
    const state_t positive_state = make_pair(state.first + 1, state.second);
    const state_t negative_state = make_pair(state.first, state.second + 1);
    return make_pair(make_pair(positive_state, positiveprob), make_pair(negative_state, negativeprob));
}

/// \return number of steps to the horizon
/// horizon = is one state
inline int steps_to_end(uint horizon, state_t state){
    return 2 + horizon - (state.first + state.second);
}

int main(){
    // -- initialize ---------------------------------------------------
    // number of steps (horizon = 1 is 1 state)
    const uint horizon = 1000;
    // discretization of gittins value
    const double lambda_step = 0.01;
    // discount factor
    const double gamma = 0.99;
    // output file name
    const string output_filename = "gittins.csv";

    cout << "Horizon: " << horizon << endl;
    cout << "Lambda precision: " << lambda_step << endl;
    cout << "Discount factor: " << gamma << endl;

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

    //cout << "States:" << endl << states << endl;

    // state map
    std::cout << "Constructing state map ..." << endl;
    map<state_t, int> state2index;
    for(size_t i = 0; i < states.size(); ++i)
        state2index[states[i]] = i;

    // -- generate lambdas -----------------------------------------
    cout << "Generating lambdas ..." << endl;
    vector<double> lambdas;
    lambdas.reserve(ceil(1.0/lambda_step));

    for(double lambda = 0.0; lambda <= 1.0; lambda+=lambda_step)
        lambdas.push_back(lambda);

    const size_t lambda_count = lambdas.size();
    cout << "Number of lambdas: " << lambda_count << endl;

    //cout << "Lambdas: " << endl;
    //cout << lambdas << endl;

    // ------------------------------------------
    // allocate the state value function: lambda_index, stateindex
    cout << "Computing value function ..." << endl;

    // value function
    Eigen::MatrixXd valuefunction(lambda_count, state_count);
    // difference between the certain an uncertain options
    Eigen::MatrixXd differences(lambda_count, state_count);

    // iterate over each lambda separately (this step can be parallelized)
    // lambda is the value of the certain arm
    #pragma omp parallel for
    for(size_t ilambda=0; ilambda < lambda_count; ilambda++){
        // iterate over states backwards (dynamic programming)
        // state of the uncertain arm
        for(int istate=state_count-1; istate >= 0; istate--){
            auto state = states[istate];
            auto nextstates = transition(state);
            pair<state_t, double>
                positive_sp = nextstates.first,
                negative_sp = nextstates.second;

            // value of taking the uncertain function
            // states beyond the horizon have value 0
            auto value_uncertain =
                positive_sp.second * (1.0 + (steps_to_end(horizon, positive_sp.first) <= 0 ?
                                0 : gamma*valuefunction(ilambda, state2index[positive_sp.first]))) +
                negative_sp.second * (0.0 + (steps_to_end(horizon, negative_sp.first) <= 0 ?
                                0 : gamma*valuefunction(ilambda, state2index[negative_sp.first])));

            // compute value of the certain option
            // discount = 1 must be treated differently
            auto value_certain = (gamma < 1.0) ?
                    lambdas[ilambda] * (1 - pow(gamma,steps_to_end(horizon, state))) / (1-gamma) :
                    lambdas[ilambda] * steps_to_end(horizon, state);

            valuefunction(ilambda,istate) = max(value_uncertain, value_certain);
            differences(ilambda,istate) = abs(value_uncertain - value_certain);
        }
    }

    // -- compute indices -------------------------------------------------
    cout << "Computing errors ... " << endl;
    size_t maxerror_state;
    auto maxerror = differences.colwise().minCoeff().maxCoeff(&maxerror_state);

    // -- compute indices -------------------------------------------------
    cout << "Computing indices ... " << endl;
    vector<double> results(state_count);
    #pragma omp parallel for
    for(size_t i = 0; i < state_count; i++) {
        size_t x;
        differences.col(i).minCoeff(&x);
        results[i] = lambdas[x];
    }

    auto end = chrono::steady_clock::now();
    auto diff = end - start;

    cout << "Maximum error: " << maxerror << " for state " << states[maxerror_state] << endl;
    //cout << "Value function:\n" << valuefunction << endl;

    cout << "Duration (computations): " << chrono::duration <double, milli> (diff).count() << " ms" << endl;
    cout << "Writing results to a csv: " << output_filename << endl;
    ofstream fout(output_filename);
    fout << "Positive,Negative,Index" << endl;
    for (size_t i = 0; i < state_count; i++) {
        fout << states[i].first << "," << states[i].second << "," << results[i] << endl;
    }

    return 0;
}
