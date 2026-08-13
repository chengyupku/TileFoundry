extern "C" __global__ void tilefoundry_copy_latency(
    const unsigned char* src,
    unsigned char* dst,
    int nbytes,
    int repetitions) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int r = 0; r < repetitions; ++r) {
    for (int i = 0; i < nbytes; ++i) {
      if (r == 0) {
        reinterpret_cast<volatile unsigned char*>(dst)[i] =
            reinterpret_cast<volatile const unsigned char*>(src)[i];
      } else {
        reinterpret_cast<volatile unsigned char*>(dst)[i] =
            reinterpret_cast<volatile const unsigned char*>(dst)[i];
      }
    }
    __threadfence_block();
  }
}

extern "C" __global__ void tilefoundry_copy_ii(
    const unsigned char* src,
    unsigned char* dst,
    int nbytes,
    int repetitions,
    int independent_chains) {
  int chain = static_cast<int>(threadIdx.x);
  if (chain < independent_chains) {
    const unsigned char* chain_src = src + chain * nbytes;
    unsigned char* chain_dst = dst + chain * nbytes;
    for (int r = 0; r < repetitions; ++r) {
      for (int i = 0; i < nbytes; ++i) {
        reinterpret_cast<volatile unsigned char*>(chain_dst)[i] =
            reinterpret_cast<volatile const unsigned char*>(chain_src)[i];
      }
    }
  }
}
