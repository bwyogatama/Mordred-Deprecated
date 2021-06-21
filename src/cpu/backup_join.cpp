int main(int argc, char** argv)
{
  // int num_fact       = 1<<28;
  // int num_dim      = 1<<16;
  int num_fact       = 256 * 1<<20;
  int num_dim      = 16 * 1<<20;
  int num_trials     = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", num_fact);
  args.GetCmdLineArgument("d", num_dim);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
      "[--n=<num fact>] "
      "[--d=<num dim>] "
      "[--t=<num trials>] "
      "[--device=<device-id>] "
      "[--v] "
      "\n", argv[0]);
    exit(0);
  }

  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

  for (int t = 0; t < num_trials; t++) {
    float time_join1_gpu, time_join2_gpu;

    time_probe1 = joinScalar(d_dim_key, d_dim_val, d_fact_fkey, d_fact_val, num_dim, num_fact, g_allocator);
    time_probe2 = joinSIMD(d_dim_key, d_dim_val, d_fact_fkey, d_fact_val, num_dim, num_fact, g_allocator);

    cout<< "{"
      << "\"num_dim\":" << num_dim
      << ",\"num_fact\":" << num_fact
      << ",\"time_build\":" << time_build
      << ",\"time_probe1\":" << time_probe1
      << ",\"time_probe2\":" << time_probe2
      << "}" << endl;
  }

  return 0;
}


