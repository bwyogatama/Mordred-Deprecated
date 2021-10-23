//----- Include files -------------------------------------------------------
#include <assert.h>             // Needed for assert() macro
#include <stdio.h>              // Needed for printf()
#include <stdlib.h>             // Needed for exit() and ato*()
#include <math.h>               // Needed for pow()
#include <string>
#include <unistd.h>
#include <iostream>

//----- Constants -----------------------------------------------------------
#define  FALSE          0       // Boolean false
#define  TRUE           1       // Boolean true

//----- Function prototypes -------------------------------------------------
int      zipf(double alpha, int n);  // Returns a Zipf random variable
double   rand_val(int seed);         // Jain's RNG

using namespace std;

//===== Main program ========================================================
int main()
{
  double alpha;                 // Alpha parameter
  int n;                     // N parameter
  int    num_values;            // Number of values
  int    zipf_rv;               // Zipf random variable
  string temp_string;

  // Prompt for random number seed and then use it
  rand_val(123);

  // Prompt for alpha value
  printf("Alpha value ========================================> ");
  cin >> temp_string;
  alpha = stof(temp_string);

  // Prompt for N value
  printf("N value ============================================> ");
  cin >> temp_string;
  n = stoi(temp_string);

  // // Generate and output zipf random variables
  for (int i=0; i<100; i++)
  {
    zipf_rv = zipf(alpha, n);
    cout << zipf_rv << endl;
  //   fprintf(fp, "%d \n", zipf_rv);
  }

  return 0;
}

/
int zipf(double alpha, int n)
{
  // static int first = TRUE;      // Static first time flag
  // static double c = 0;          // Normalization constant
  double c = 0;
  double z;                     // Uniform random number (0 < z < 1)
  double sum_prob;              // Sum of probabilities
  double zipf_value;            // Computed exponential value to be returned
  int    i;                     // Loop counter

  for (i=1; i<=n; i++)
    c = c + (1.0 / pow((double) i, alpha));
  c = 1.0 / c;

  // Pull a uniform random number (0 < z < 1)
  do
  {
    z = rand_val(0);
  }
  while ((z == 0) || (z == 1));

  // Map z to the value
  sum_prob = 0;
  for (i=1; i<=n; i++)
  {
    sum_prob = sum_prob + c / pow((double) i, alpha);
    if (sum_prob >= z)
    {
      zipf_value = i;
      break;
    }
  }

  // Assert that zipf_value is between 1 and N
  assert((zipf_value >=1) && (zipf_value <= n));

  return(zipf_value);
}

double rand_val(int seed)
{
  const long  a =      16807;  // Multiplier
  const long  m = 2147483647;  // Modulus
  const long  q =     127773;  // m div a
  const long  r =       2836;  // m mod a
  static long x;               // Random int value
  long        x_div_q;         // x divided by q
  long        x_mod_q;         // x modulo q
  long        x_new;           // New x value

  // Set the seed if argument is non-zero and then return zero
  if (seed > 0)
  {
    x = seed;
    return(0.0);
  }

  // RNG using integer arithmetic
  x_div_q = x / q;
  x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  if (x_new > 0)
    x = x_new;
  else
    x = x_new + m;

  // Return a random value between 0.0 and 1.0
  return((double) x / m);
}