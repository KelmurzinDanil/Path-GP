#include <vector>
#include <cmath>
#include <iostream>
#include <array>
#include <stdexcept>
#include <functional>
#include <limits>
#include <omp.h>
#include <map>
#include <mutex>
#include <atomic>
#include <memory>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>

namespace py = pybind11;

enum class TokenType {
    Variable,     // переменная
    Constant,     // фикс. константа
    Placeholder,  // оптим. константа
    Add,          // +
    Sub,          // -
    Mul,          // *
    Div,          // /
    Pow,          // ^ 
    Sin,          // sin()
    Cos,          // cos()
    Tan,          // tan()
    Abs,          // abs()
    Sqrt,         // sqrt()
    Log,          // log()
    Exp           // exp()
};

struct Token {
    TokenType type;
    double value;
};


template <typename T, size_t MaxSize = 32>
class Stack{
    private:
        std::array<T, MaxSize> data;
        size_t top_ptr = 0;

    public: 
    void clear() noexcept{
        top_ptr = 0;
    }

    // (!) нужно следить за копированием, чтобы не полетело std::terminate()
    void push(T val) noexcept{
        data[top_ptr++] = val;
    }

    T pop() noexcept{
        return data[--top_ptr];
    }

    size_t size() const noexcept{
        return top_ptr;
    }

    bool is_empty() const noexcept{
        return top_ptr == 0;
    }
};


double evaluate_rpn(const std::vector<Token>& expression, const std::vector<double>& x_values) {
    Stack<double> stack; 

    for (const auto& token : expression) {
        switch (token.type) {
            case TokenType::Variable: {
                size_t index = static_cast<size_t>(token.value);
                stack.push(x_values[index]);
                break;
            }
            case TokenType::Constant:
            case TokenType::Placeholder: { 
                stack.push(token.value);
                break;
            }
            case TokenType::Add: {
                double a1 = stack.pop();
                double a2 = stack.pop();
                stack.push(a2 + a1);
                break;
            }
            case TokenType::Sub: {
                double a1 = stack.pop();
                double a2 = stack.pop();
                stack.push(a2 - a1);
                break;
            }
            case TokenType::Mul: {
                double a1 = stack.pop();
                double a2 = stack.pop();
                stack.push(a2 * a1);
                break;
            }
            case TokenType::Div: {
                double a1 = stack.pop();
                double a2 = stack.pop();
                if (std::abs(a1) < 1e-9) { 
                    return std::nan("");
                }
                stack.push(a2 / a1);
                break;
            }
            case TokenType::Sin: {
                double a = stack.pop();
                stack.push(std::sin(a));
                break;
            }
            case TokenType::Cos: {
                double a = stack.pop();
                stack.push(std::cos(a));
                break;
            }
            case TokenType::Log: {
                double a = stack.pop();
                if (a < 1e-9) { 
                    return std::nan("");
                }
                stack.push(std::log(a));
                break;
            }
            case TokenType::Exp: {
                double a = stack.pop();
                double res = std::exp(a);
                if (!std::isfinite(res)) {
                    return std::nan("");
                }
                stack.push(res);
                break;
            }
            case TokenType::Tan: {
                double a = stack.pop();
                double res = std::tan(a);
                if (!std::isfinite(res)) {
                    return std::nan("");
                }
                stack.push(res);
                break;
            }
            case TokenType::Abs: {
                double a = stack.pop();
                stack.push(std::abs(a));
                break;
            }
            case TokenType::Sqrt: {
                double a = stack.pop();
                if (a < 0) {
                    return std::nan("");
                }
                stack.push(std::sqrt(a));
                break;
            }
            case TokenType::Pow: {
                double a1 = stack.pop(); 
                double a2 = stack.pop(); 
                if (a2 < 0 && std::abs(a1 - std::round(a1)) > 1e-9) {
                    return std::nan("");
                }
                double res = std::pow(a2, a1);
                if (!std::isfinite(res)) {
                    return std::nan("");
                }
                stack.push(res);
                break;
            }
            default:
                return std::nan("");
        }
    }

    if (stack.is_empty()) {
        return std::nan("");
    }

    return stack.pop();
}

bool is_binary(TokenType type) noexcept {
    switch (type) {
        case TokenType::Add:
        case TokenType::Sub:
        case TokenType::Mul:
        case TokenType::Div:
        case TokenType::Pow:{
            return true;
        }
        default:
            return false;
    }
}
bool is_unary(TokenType type) noexcept {
    switch (type) {
        case TokenType::Sin:
        case TokenType::Cos:
        case TokenType::Tan:
        case TokenType::Abs:
        case TokenType::Sqrt:
        case TokenType::Log:
        case TokenType::Exp:
            return true;
        default:
            return false;
    }
}
bool is_operand(TokenType type) noexcept {
    return type == TokenType::Variable || 
           type == TokenType::Constant || 
           type == TokenType::Placeholder;
}

class FormulaGenerator {
    private:
        int target_length;                   // целевая длина формулы (L)
        std::vector<Token> allowed_tokens;   // набор токенов

        // вызывается, когда найдена полностью валидная формула
        std::function<void(const std::vector<Token>&)> on_formula_found;

        void generate_step(int current_step, int stack_size, std::vector<Token>& current_expression) {
            if (current_step == target_length) {
                if (stack_size == 1) {
                    on_formula_found(current_expression);
                }
                return;
            }

            for (const auto& token : allowed_tokens) {
                int next_stack_size = stack_size;

                if(is_binary(token.type) == true){
                    if (stack_size < 2) {
                        continue; 
                    }
                    next_stack_size = stack_size - 1;
                } 
                else if (is_unary(token.type) == true){
                    if (stack_size < 1) {
                        continue; 
                    }
                }
                else if(is_operand(token.type)){
                    next_stack_size = stack_size + 1;
                }

                if(next_stack_size < 1){
                    continue;
                }

                current_expression.push_back(token); 
                
                generate_step(current_step + 1, next_stack_size, current_expression); 
                
                current_expression.pop_back(); 
                
            }
        }

    public:
        FormulaGenerator(int length, 
                        const std::vector<Token>& allowed, 
                        std::function<void(const std::vector<Token>&)> callback)
            : target_length(length), allowed_tokens(allowed), on_formula_found(callback) {
        }

        void start() {
            #pragma omp parallel for
            for (int i = 0; i < (int)allowed_tokens.size(); ++i) {
                const auto& root_token = allowed_tokens[i];
                if (root_token.type == TokenType::Variable || 
                    root_token.type == TokenType::Constant || 
                    root_token.type == TokenType::Placeholder){
                        
                    std::vector<Token> local_expression;
                    local_expression.reserve(target_length);
                    local_expression.push_back(root_token);

                    generate_step(1, 1, local_expression);
                }
            }
        }
};

struct ParetoRecord {
    std::vector<Token> expression;
    double mse = std::numeric_limits<double>::max(); 
    bool is_valid = false;     

    bool operator<(const ParetoRecord& other) const {
        return mse < other.mse;
    }
};

class ParetoFrontier{
    private: 
        size_t max_complexity;
        size_t K_best;
    
        std::unique_ptr<std::atomic<double>[]> best_mses;
        std::vector<std::vector<ParetoRecord>> frontier;
        std::vector<omp_lock_t> locks;

    public:
        ParetoFrontier(size_t max_comp = 32, size_t k_best_val = 50) : max_complexity(max_comp), K_best(k_best_val)  {
            frontier.resize(max_complexity);
            locks.resize(max_complexity);
            best_mses = std::make_unique<std::atomic<double>[]>(max_complexity);

            for (size_t i = 0; i < max_complexity; ++i) {
                omp_init_lock(&locks[i]); 
                best_mses[i].store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
            }
        }

        ~ParetoFrontier() {
            for (size_t i = 0; i < max_complexity; ++i) {
                omp_destroy_lock(&locks[i]); 
            }
        }
        
        std::vector<ParetoRecord> get_expressions(int complexity) const {
            return frontier[complexity];
        }

        size_t get_max_complexity() const{
            return max_complexity;
        }

        bool update(int complexity, const std::vector<Token>& expression, double mse) {
            if (mse >= get_best_mse(complexity)) {
                return false;
            }

            omp_set_lock(&locks[complexity]);

            auto& list = frontier[complexity];
            double current_worst_mse = list.empty() || list.size() < K_best ? 
                std::numeric_limits<double>::max() : list.back().mse;

            if (mse < current_worst_mse){
                ParetoRecord new_record{expression, mse, true};
            
                auto it = std::upper_bound(list.begin(), list.end(), new_record, 
                    [](const ParetoRecord& a, const ParetoRecord& b) {
                        return a.mse < b.mse;
                    }
                );
                list.insert(it, new_record);

                if (list.size() > K_best) {
                    list.pop_back();
                }

                double new_gatekeeper = (list.size() < K_best) ? 
                    std::numeric_limits<double>::max() : list.back().mse;

                best_mses[complexity].store(new_gatekeeper, std::memory_order_release);

                omp_unset_lock(&locks[complexity]);
                return true;
            }

            omp_unset_lock(&locks[complexity]); 
            return false;
        }
        double get_best_mse(int complexity) const{
            return best_mses[complexity].load(std::memory_order_relaxed);
        }

        void print() const {
        std::cout << "\n=== Pareto Frontier (Top-" << K_best << " per complexity) ===\n";
        for (size_t i = 0; i < max_complexity; ++i) {
            const auto& list = frontier[i];
            if (!list.empty()) {
                std::cout << "Complexity: " << i << " (" << list.size() << " formulas found):\n";
                for (size_t rank = 0; rank < list.size(); ++rank) {
                    std::cout << "  Rank " << rank + 1 << " | MSE: " << list[rank].mse 
                              << " | Tokens: " << list[rank].expression.size() << "\n";
                }
            }
        }
    }
};

class LMOptimizer {
    private:
        std::vector<double> calculate_residuals(
            const std::vector<Token>& expression,
            const std::vector<std::vector<double>>& X,
            const std::vector<double>& Y,
            const std::vector<double>& params){
                size_t n = X.size();
                std::vector<double> r(n);
                auto bound_expr = bind_parameters(expression, params);

                for (size_t i = 0; i < n; ++i){
                    double pred = evaluate_rpn(bound_expr, X[i]);
                    r[i] = std::isnan(pred) || std::isinf(pred) ? 1e10 : (pred - Y[i]);
                }

            return r;
        }
        
        std::vector<double> solve_system(const std::vector<double>& H, const std::vector<double>& g, size_t K) {
            std::vector<double> delta(K, 0.0);
            
            // Свободный член b = -g
            std::vector<double> b(K);
            for (size_t i = 0; i < K; ++i) {
                b[i] = -g[i];
            }

            if (K == 1) {
                double det = H[0];
                if (std::abs(det) < 1e-12) return {}; // Матрица вырождена
                delta[0] = b[0] / det;
                return delta;
            } 
            else if (K == 2) {
                // Матрица H:
                // [ H0  H1 ]
                // [ H2  H3 ] 
                double det = H[0] * H[3] - H[1] * H[2];
                if (std::abs(det) < 1e-12) return {};

                // Заменяем столбцы на вектор b
                double det_0 = b[0] * H[3] - H[1] * b[1];
                double det_1 = H[0] * b[1] - b[0] * H[2];

                delta[0] = det_0 / det;
                delta[1] = det_1 / det;
                return delta;
            } 
            else if (K == 3) {
                // Матрица H:
                // [ H0  H1  H2 ]
                // [ H3  H4  H5 ]
                // [ H6  H7  H8 ]
                double det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
                            H[1] * (H[3] * H[8] - H[5] * H[6]) +
                            H[2] * (H[3] * H[7] - H[4] * H[6]);

                if (std::abs(det) < 1e-12) return {};

                // det_0 (заменяем 0-й столбец на b)
                double det_0 = b[0] * (H[4] * H[8] - H[5] * H[7]) -
                            H[1] * (b[1] * H[8] - H[5] * b[2]) +
                            H[2] * (b[1] * H[7] - H[4] * b[2]);

                // det_1 (заменяем 1-й столбец на b)
                double det_1 = H[0] * (b[1] * H[8] - H[5] * b[2]) -
                            b[0] * (H[3] * H[8] - H[5] * H[6]) +
                            H[2] * (H[3] * b[2] - b[1] * H[6]);

                // det_2 (заменяем 2-й столбец на b)
                double det_2 = H[0] * (H[4] * b[2] - b[1] * H[7]) -
                            H[1] * (H[3] * b[2] - b[1] * H[6]) +
                            b[0] * (H[3] * H[7] - H[4] * H[6]);

                delta[0] = det_0 / det;
                delta[1] = det_1 / det;
                delta[2] = det_2 / det;
                return delta;
            }

            return {}; 
        }
        
    public:
        static std::vector<Token> bind_parameters(std::vector<Token> expr, const std::vector<double>& params) {
            size_t p_idx = 0;
            for (auto& token : expr) {
                if (token.type == TokenType::Placeholder) {
                    if (p_idx < params.size()) {
                        token.value = params[p_idx++];
                    }
                }
            }
            return expr;
        }

        std::vector<double> optimize(
            const std::vector<Token>& expression,
            const std::vector<std::vector<double>>& X,
            const std::vector<double>& Y,
            std::vector<double> start_params) {
                size_t K = start_params.size();
                size_t N = X.size();

                double lambda = 0.001;          
                double eps = 1e-8;              
                std::vector<double> params = start_params;

                for (int iter = 0; iter < 50; ++iter) { 
                    auto r = calculate_residuals(expression, X, Y, params);
                    double sse_old = 0.0;
                    for (double val : r) sse_old += val * val;
                    std::vector<double> J(K * N, 0.0);
                    std::vector<double> g(K, 0.0);
                    std::vector<double> H(K * K, 0.0);

                    

                    for (size_t i = 0; i < K; ++i) {
                        auto temp_params = params;
                        temp_params[i] += eps;
                        auto r_perturbed = calculate_residuals(expression, X, Y, temp_params);

                        for (size_t j = 0; j < N; ++j) {
                            J[i * N + j] = (r_perturbed[j] - r[j]) / eps;
                        }
                    }
                    for (size_t i = 0; i < K; ++i) {
                        double sum_g = 0.0;
                        for (size_t j = 0; j < N; ++j) {
                            sum_g += J[i * N + j] * r[j];
                        }
                        g[i] = sum_g;

                        for (size_t j = i; j < K; ++j) { 
                            double sum_h = 0.0;
                            for (size_t k = 0; k < N; ++k) {
                                sum_h += J[i * N + k] * J[j * N + k]; 
                            }
                            
                            H[i * K + j] = sum_h;
                            
                            if (i != j) {
                                H[j * K + i] = sum_h;
                            }
                            else {
                                H[i * K + i] = sum_h + lambda; 
                            }
                        }
                    }
                    
                    std::vector<double> delta_params = solve_system(H, g, K);
                    if (delta_params.empty()) {
                        lambda *= 10.0;
                        continue;
                    }

                    std::vector<double> next_params(K);

                    for(int i = 0; i < K; ++i){
                        next_params[i] = params[i] + delta_params[i];
                    }

                    auto r_next = calculate_residuals(expression, X, Y, next_params);
                    double sse_new = 0.0;
                    for (double val : r_next) sse_new += val * val;

                    if(sse_new < sse_old){
                        lambda /= 10.0;
                        params = next_params;
                    }
                    else{
                        lambda *= 10.0;
                    }
                    
                }
            return params;
        }
                
};


class Evaluator {
private:
    std::vector<std::vector<double>> X_data; // массив входных фич
    std::vector<double> Y_data;              // массив целевых значений
    ParetoFrontier& frontier;                // ссылка на Парето-фронт
    LMOptimizer optimizer;

    int count_placeholders(const std::vector<Token>& expression) const {
        int count = 0;
        for (const auto& t : expression){
            if(t.type == TokenType::Placeholder) count++;
        }
        return count;
    }

public:
    Evaluator(const std::vector<std::vector<double>>& X, 
              const std::vector<double>& Y, 
              ParetoFrontier& front)
        : X_data(X), Y_data(Y), frontier(front) {}

    void evaluate_formula(const std::vector<Token>& expression) {
    int n_placeholders = count_placeholders(expression);
    int placeholder_penalty = 1; 
    int complexity = expression.size() + (n_placeholders * placeholder_penalty);

    size_t max_complexity = frontier.get_max_complexity(); 
    if (complexity >= (int)max_complexity) {
        return; 
    }
    
    std::vector<Token> optimized_expression = expression;
    double final_mse = std::numeric_limits<double>::max();
    size_t n_samples = X_data.size();
    
    auto calc_ols_mse = [&](const std::vector<Token>& expr) -> double {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        std::vector<double> preds(n_samples);

        for (size_t i = 0; i < n_samples; ++i) {
            double pred = evaluate_rpn(expr, X_data[i]);
            if (std::isnan(pred) || std::isinf(pred)) return std::numeric_limits<double>::max();
            
            preds[i] = pred;
            sum_x += pred;
            sum_y += Y_data[i];
            sum_xy += pred * Y_data[i];
            sum_x2 += pred * pred;
        }

        double mean_x = sum_x / n_samples;
        double mean_y = sum_y / n_samples;
        double variance_x = (sum_x2 / n_samples) - (mean_x * mean_x);
        double cov_xy = (sum_xy / n_samples) - (mean_x * mean_y);

        double a = 1.0;
        double b = 0.0;

        if (variance_x > 1e-12) {
            a = cov_xy / variance_x;
            b = mean_y - a * mean_x;
        } else {
            a = 0.0;
            b = mean_y; 
        }

        double mse = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            double diff = Y_data[i] - (a * preds[i] + b);
            mse += diff * diff;
        }
        return mse / n_samples;
    };

    if (n_placeholders > 0) {
        std::vector<std::vector<double>> start_points = {
            std::vector<double>(n_placeholders, 1.0),   
            std::vector<double>(n_placeholders, -1.0),  
            std::vector<double>(n_placeholders, 0.1)   
        };

        std::vector<double> best_params;

        for (const auto& start_p : start_points) {
            auto opt_p = optimizer.optimize(expression, X_data, Y_data, start_p);
            auto test_expr = LMOptimizer::bind_parameters(expression, opt_p);
            
            double test_mse = calc_ols_mse(test_expr);

            if (test_mse < final_mse) {
                final_mse = test_mse;
                best_params = opt_p;
            }
        }
        optimized_expression = LMOptimizer::bind_parameters(expression, best_params);
    } 
    else {
        double test_mse = calc_ols_mse(expression);
        
        if (test_mse > frontier.get_best_mse(complexity)) {
            return;
        }
        final_mse = test_mse;
    }

    frontier.update(complexity, optimized_expression, final_mse);
}
};

std::string tokens_to_string(const std::vector<Token>& expr){
    std::string res = "";
    for (const auto& t : expr) {
        if (t.type == TokenType::Variable) res += "x" + std::to_string((int)t.value) + " ";
        else if (t.type == TokenType::Constant) res += std::to_string(t.value) + " ";
        else if (t.type == TokenType::Placeholder) res += std::to_string(t.value) + " ";
        else if (t.type == TokenType::Add) res += "+ ";
        else if (t.type == TokenType::Sub) res += "- ";
        else if (t.type == TokenType::Mul) res += "* ";
        else if (t.type == TokenType::Div) res += "/ ";
        else if (t.type == TokenType::Pow) res += "^ ";
        else if (t.type == TokenType::Sin) res += "sin ";
        else if (t.type == TokenType::Cos) res += "cos ";
        else if (t.type == TokenType::Tan) res += "tan ";
        else if (t.type == TokenType::Abs) res += "abs ";
        else if (t.type == TokenType::Sqrt) res += "sqrt ";
        else if (t.type == TokenType::Exp) res += "exp ";
        else if (t.type == TokenType::Log) res += "log ";
    }
    return res;
}


struct PyParetoRecord {
    int complexity;
    double mse;
    std::string expression_str;
};

std::vector<PyParetoRecord> run_brute_force(
    const std::vector<std::vector<double>>& X, 
    const std::vector<double>& Y, 
    int max_length,
    const std::vector<double>& allowed_constants,
    bool optimize_constants,
    const std::vector<std::string>& allowed_ops, 
    size_t k_best) {
        std::vector<Token> alphabet;

        if(!X.empty()){
            for (size_t i = 0; i < X[0].size(); ++i) {
            alphabet.push_back({TokenType::Variable, (double)i});
            }
        }

        for (double c: allowed_constants){
            alphabet.push_back({TokenType::Constant, c});
        }

        if (optimize_constants) {
            alphabet.push_back({TokenType::Placeholder, 0.0}); 
        }

        static const std::map<std::string, TokenType> op_map = {
            {"add", TokenType::Add},
            {"sub", TokenType::Sub},
            {"mul", TokenType::Mul},
            {"div", TokenType::Div},
            {"pow", TokenType::Pow},
            {"sin", TokenType::Sin},
            {"cos", TokenType::Cos},
            {"tan", TokenType::Tan},
            {"abs", TokenType::Abs},
            {"sqrt", TokenType::Sqrt},
            {"log", TokenType::Log},
            {"exp", TokenType::Exp}
        };

        for (const auto& op_name : allowed_ops) {
            auto it = op_map.find(op_name);
            if (it != op_map.end()) {
                alphabet.push_back({it->second, 0.0});
            } else {
                std::cerr << "[Warning] Unknown operator name passed from Python: " << op_name << std::endl;
            }
        }

        ParetoFrontier frontier(max_length + 1, k_best);
        Evaluator evaluator(X, Y, frontier);

        for (int len = 1; len <= max_length; ++len) {
            FormulaGenerator generator(
                len, 
                alphabet,
                [&evaluator](const std::vector<Token>& expr){
                    evaluator.evaluate_formula(expr);
                }
            );
            generator.start(); 
        }


        std::vector<PyParetoRecord> results;
        for (int i = 1; i <= max_length; ++i) {
            auto expr_list = frontier.get_expressions(i);
            
            for (const auto& record : expr_list) {
                results.push_back({
                    i, 
                    record.mse, 
                    tokens_to_string(record.expression)
                });
            }
        }

        return results;
        }

PYBIND11_MODULE(fast_symbolic, m) {
    m.doc() = "C++ Brute-force Symbolic Regression Engine";

    py::class_<PyParetoRecord>(m, "ParetoRecord")
        .def_readonly("complexity", &PyParetoRecord::complexity)
        .def_readonly("mse", &PyParetoRecord::mse)
        .def_readonly("expression", &PyParetoRecord::expression_str);

    m.def("run_brute_force", &run_brute_force, 
          "Run symbolic regression brute force with custom constants",
          py::arg("X"), py::arg("Y"), py::arg("max_length"),
          py::arg("allowed_constants") = std::vector<double>{1.0, 2.0},
          py::arg("optimize_constants") = true,
          py::arg("allowed_ops") = std::vector<std::string>{"add", "sub", "mul", "div", "sin", "cos", "exp", "log"},
          py::arg("k_best") = 50);
}