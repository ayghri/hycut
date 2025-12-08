#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>   // For atof, atoi
#include <stdexcept> // For std::invalid_argument, std::out_of_range
#include <chrono>    // For timing
#include <iomanip>   // For std::fixed, std::setprecision
#include <cmath>     // For lgamma, exp, log, pow, fabs, sqrt
#include <numeric>   // (was intended for accumulate etc.)
#include <limits>    // For std::numeric_limits
#include <random>    // For generating random z values

// Function to calculate 2F1(a_int, b; c; z) using stable methods
// a_int: negative integer
// b, c: doubles, c > b > 0
// z: double, z in [0, 1]
double hypergeometric_2F1_stable(int a_int, double b, double c, double z)
{
    if (a_int > 0)
    {
        throw std::invalid_argument("'a' (a_int) must be a non-positive integer for this implementation.");
    }
    if (a_int == 0)
    {
        return 1.0;
    }

    constexpr double epsilon = 1e-14;

    if (std::fabs(z) < epsilon)
    {
        return 1.0;
    }

    double m_double = static_cast<double>(-a_int);

    if (std::fabs(z - 1.0) < epsilon)
    {
        double log_gamma_c = std::lgamma(c);
        double log_gamma_c_minus_b_plus_m = std::lgamma(c - b + m_double);
        double log_gamma_c_plus_m = std::lgamma(c + m_double);
        double log_gamma_c_minus_b = std::lgamma(c - b);
        return std::exp(log_gamma_c + log_gamma_c_minus_b_plus_m - log_gamma_c_plus_m - log_gamma_c_minus_b);
    }

    static const double PFAFF_KUMMER_THRESHOLD = (3.0 - std::sqrt(5.0)) / 2.0; // Approx 0.381966

    double sum_val = 1.0;
    double term = 1.0;
    double a_param_double = static_cast<double>(a_int);

    if (z < PFAFF_KUMMER_THRESHOLD)
    {
        double c_minus_b = c - b;
        double z_pfaff = z / (z - 1.0);

        for (int k_idx = 0; k_idx < static_cast<int>(m_double); ++k_idx)
        {
            double k_val = static_cast<double>(k_idx);
            term *= (a_param_double + k_val) * (c_minus_b + k_val) * z_pfaff;
            term /= ((c + k_val) * (k_val + 1.0));
            sum_val += term;
        }
        return std::pow(1.0 - z, m_double) * sum_val;
    }
    else
    {
        double C_prime = a_param_double + b - c + 1.0;
        double z_kummer = 1.0 - z;

        for (int k_idx = 0; k_idx < static_cast<int>(m_double); ++k_idx)
        {
            double k_val = static_cast<double>(k_idx);

            double num_factor = (a_param_double + k_val) * (b + k_val) * z_kummer;
            double den_factor = (C_prime + k_val) * (k_val + 1.0);

            if (std::fabs(den_factor) < epsilon * epsilon)
            {
                if (std::fabs(num_factor) < epsilon * epsilon)
                {
                    term = 0.0;
                }
                else
                {
                    term = std::copysign(std::numeric_limits<double>::infinity(), num_factor / den_factor);
                }
            }
            else
            {
                term *= num_factor / den_factor;
            }
            sum_val += term;
            if (!std::isfinite(term) && term != 0.0)
                break;
        }

        double log_gamma_c_minus_b_plus_m = std::lgamma(c - b + m_double);
        double log_gamma_c = std::lgamma(c);
        double log_gamma_c_minus_b = std::lgamma(c - b);
        double log_gamma_c_plus_m = std::lgamma(c + m_double);
        double prefactor = std::exp(log_gamma_c_minus_b_plus_m + log_gamma_c - log_gamma_c_minus_b - log_gamma_c_plus_m);

        return prefactor * sum_val;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <a> <b> <c> <num_z_elements> <n_iterations>" << std::endl;
        std::cerr << "  a: negative integer (e.g., -10)" << std::endl;
        std::cerr << "  b: positive double" << std::endl;
        std::cerr << "  c: positive double, c > b" << std::endl;
        std::cerr << "  num_z_elements: number of z values to process (e.g., 10000)" << std::endl;
        std::cerr << "  n_iterations: positive integer, number of benchmark iterations over the z vector" << std::endl;
        return 1;
    }

    int a_val;
    double b_val, c_val;
    int num_z_elements;
    long long n_iterations; // Use long long for n_iterations

    try
    {
        a_val = std::stoi(argv[1]);
        b_val = std::stod(argv[2]);
        c_val = std::stod(argv[3]);
        num_z_elements = std::stoi(argv[4]);
        n_iterations = std::stoll(argv[5]); // stoll for long long
    }
    catch (const std::invalid_argument &ia)
    {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return 1;
    }
    catch (const std::out_of_range &oor)
    {
        std::cerr << "Argument out of range: " << oor.what() << std::endl;
        return 1;
    }

    // Validate constraints
    if (a_val >= 0)
    {
        std::cerr << "Error: 'a' must be a negative integer." << std::endl;
        return 1;
    }
    if (b_val <= 0)
    {
        std::cerr << "Error: 'b' must be positive." << std::endl;
        return 1;
    }
    if (c_val <= 0)
    {
        std::cerr << "Error: 'c' must be positive." << std::endl;
        return 1;
    }
    if (c_val <= b_val)
    {
        std::cerr << "Error: 'c' must be greater than 'b'." << std::endl;
        return 1;
    }
    if (num_z_elements <= 0)
    {
        std::cerr << "Error: 'num_z_elements' must be positive." << std::endl;
        return 1;
    }
    if (n_iterations <= 0)
    {
        std::cerr << "Error: 'n_iterations' must be positive." << std::endl;
        return 1;
    }

    std::cout << "CPU Hypergeometric 2F1 Benchmark (Vectorized Z)" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  a = " << a_val << std::endl;
    std::cout << "  b = " << b_val << std::endl;
    std::cout << "  c = " << c_val << std::endl;
    std::cout << "  Number of z elements = " << num_z_elements << std::endl;
    std::cout << "  Benchmark iterations (over z vector) = " << n_iterations << std::endl;

    // --- Host data ---
    std::vector<double> z_values_host(num_z_elements);
    std::vector<double> results_host(num_z_elements); // To store results of one pass for verification

    // Generate random z values
    std::mt19937 rng(std::random_device{}()); // Seed with random_device
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < num_z_elements; ++i)
    {
        z_values_host[i] = dist(rng);
    }
    // Example: set specific values for testing if needed
    // if (num_z_elements > 0) z_values_host[0] = 0.0;
    // if (num_z_elements > 1) z_values_host[1] = 1.0;
    // if (num_z_elements > 2) z_values_host[2] = 0.5;

    // Perform a single pass to get results for verification (and warm-up)
    std::cout << "\nPerforming a single pass for verification..." << std::endl;
    for (int i = 0; i < num_z_elements; ++i)
    {
        try
        {
            results_host[i] = hypergeometric_2F1_stable(a_val, b_val, c_val, z_values_host[i]);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during verification calculation for z[" << i << "]: " << e.what() << std::endl;
            results_host[i] = std::numeric_limits<double>::quiet_NaN(); // Mark as NaN
        }
    }

    // Print a few results for verification
    std::cout << "Sample results (first up to 5):" << std::endl;
    std::cout << std::fixed << std::setprecision(10);
    for (int i = 0; i < std::min(num_z_elements, 5); ++i)
    {
        std::cout << "z[" << i << "] = " << z_values_host[i]
                  << ", _2F_1(...) = " << results_host[i] << std::endl;
    }
    if (num_z_elements > 5)
    {
        std::cout << "..." << std::endl;
    }

    // Timing
    std::cout << "\nStarting benchmark (" << n_iterations << " iterations over " << num_z_elements << " z values)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (long long iter = 0; iter < n_iterations; ++iter)
    {
        for (int i = 0; i < num_z_elements; ++i)
        {
            // Re-assign to a volatile variable to prevent optimizer from removing the call
            volatile double temp_result = hypergeometric_2F1_stable(a_val, b_val, c_val, z_values_host[i]);
            (void)temp_result; // Suppress unused variable warning
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

    std::cout << "Benchmark finished." << std::endl;
    if (n_iterations > 0)
    {
        double total_duration_ns = static_cast<double>(duration.count());
        double avg_time_ns_per_iteration = total_duration_ns / n_iterations;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total time for " << n_iterations << " iterations (each over " << num_z_elements << " z values): "
                  << total_duration_ns / 1e6 << " ms" << std::endl; // Total in ms

        std::cout << "Average time per iteration (processing " << num_z_elements << " elements): "
                  << avg_time_ns_per_iteration / 1e6 << " ms" << std::endl; // Avg per vector pass in ms
        std::cout << "Average time per iteration (processing " << num_z_elements << " elements): "
                  << avg_time_ns_per_iteration / 1e3 << " µs" << std::endl; // Avg per vector pass in µs

        if (num_z_elements > 0)
        {
            double total_calculations = static_cast<double>(n_iterations) * num_z_elements;
            double avg_time_ns_per_element = total_duration_ns / total_calculations;
            std::cout << "Average time per single z element computation: " << avg_time_ns_per_element << " ns" << std::endl;
            std::cout << "Average time per single z element computation: " << avg_time_ns_per_element / 1e3 << " µs" << std::endl;
        }
    }

    return 0;
}
