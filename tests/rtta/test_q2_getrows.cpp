#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstdlib>

std::vector<float> run(ggml_backend_t backend,int cols,const std::vector<unsigned char>& q,const std::vector<int32_t>& ids) {
    ggml_init_params p{16*1024*1024,nullptr,true};auto ctx=ggml_init(p);
    auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q2_K,cols,17);
    auto i=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,ids.size());
    auto out=ggml_get_rows(ctx,w,i);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buf) std::abort();
    ggml_backend_tensor_set(w,q.data(),0,q.size());ggml_backend_tensor_set(i,ids.data(),0,ids.size()*sizeof(int32_t));
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)std::abort();
    std::vector<float> result(cols*ids.size());ggml_backend_tensor_get(out,result.data(),0,result.size()*sizeof(float));
    ggml_backend_buffer_free(buf);ggml_free(ctx);return result;
}
int main(){
    auto cpu=ggml_backend_cpu_init();auto cuda=ggml_backend_cuda_init(0);if(!cuda)return 2;
    std::mt19937 rng(20260929);std::normal_distribution<float> n(0,1);
    for(int cols:{256,512,2048}) {
        std::vector<float> data(cols*17);for(auto &v:data)v=n(rng);
        std::vector<unsigned char> q(ggml_row_size(GGML_TYPE_Q2_K,cols)*17);
        ggml_quantize_chunk(GGML_TYPE_Q2_K,data.data(),q.data(),0,17,cols,nullptr);
        std::vector<int32_t> ids{16,0,7,2,7,15,1};
        auto a=run(cpu,cols,q,ids),b=run(cuda,cols,q,ids);float maxerr=0;
        for(size_t i=0;i<a.size();i++) {if(!std::isfinite(b[i]))return 3;maxerr=std::max(maxerr,std::abs(a[i]-b[i]));}
        printf("cols=%d compared=%zu max_abs_error=%.9g\n",cols,a.size(),maxerr);
        if(maxerr>1e-6)return 1;
    }
    ggml_backend_free(cuda);ggml_backend_free(cpu);puts("PASS");return 0;
}
