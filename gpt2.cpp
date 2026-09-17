#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <cmath>

struct Config {
    int32_t magic, version, n_layer, n_head, n_embd, n_vocab, n_ctx, pad;
};

struct Layer {
    const float *ln_1_w, *ln_1_b;
    const float *c_attn_w, *c_attn_b;
    const float *c_proj_w, *c_proj_b;
    const float *ln_2_w, *ln_2_b;
    const float *c_fc_w, *c_fc_b;
    const float *mlp_proj_w, *mlp_proj_b;
};

struct Model {
    Config cfg;
    const float* wte;
    const float* wpe;
    std::vector<Layer> layers;
    const float* ln_f_w;
    const float* ln_f_b;
};

template <class T>
static std::vector<T> read_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");        // read, binary
    if (!f) { perror(path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);                      // jump to end
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);                      // back to start
    std::vector<T> v(bytes / sizeof(T));        // element size
    if (fread(v.data(), 1, bytes, f) != (size_t)bytes) { perror("fread"); exit(1); }
    fclose(f);
    return v;
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; i++) { m = fmaxf(m, fabsf(a[i] - b[i])); }
    return m;
}

static void check(const char* name, const float* got, const std::string& ref_path, size_t n) {
    std::vector<float> ref = read_bin<float>(ref_path);
    if (ref.size() != n) {
        fprintf(stderr, "%s: ref has %zu floats, expected %zu\n", name, ref.size(), n);
        exit(1);
    }
    printf("%-8s max_abs_diff = %g\n", name, max_abs_diff(got, ref.data(), n));
}

static Model load_model(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    struct stat st;
    fstat(fd, &st);
    const char* base = (const char*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); exit(1); }
    close(fd);

    Model m;
    m.cfg = *(const Config*) base;
    if (m.cfg.magic != 0x47505432 || m.cfg.version != 1) {
        fprintf(stderr, "bad header: magic=0x%x, version=%d\n", m.cfg.magic, m.cfg.version);
        exit(1);
    }
    const float* p = (const float*)(base + sizeof(Config));
    m.wte = p;
    p += (size_t)(m.cfg.n_vocab * m.cfg.n_embd);
    m.wpe = p;
    p += (size_t)(m.cfg.n_ctx * m.cfg.n_embd);

    size_t C = m.cfg.n_embd;

    m.ln_f_w = nullptr; m.ln_f_b = nullptr;
    return m;
}

static void embed(const Model& m, const int32_t* tokens, int T, float* x) {
    int C = m.cfg.n_embd;
    for (int t = 0; t < T; t++) {
        const float* te = m.wte + (size_t)tokens[t] * C;
        const float* pe = m.wpe + (size_t)t * C;
        for (int i = 0; i < C; i++) x[(size_t)t * C + i] = te[i] + pe[i];
    }
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/gpt2-124M.bin";
    Model m = load_model(path);
    const Config& c = m.cfg;
    printf("n_layer=%d n_head=%d n_embd=%d n_vocab=%d n_ctx=%d\n",
        c.n_layer, c.n_head, c.n_embd, c.n_vocab, c.n_ctx);

    printf("wte[0,:5] =");
    for (int i = 0; i < 4; ++i) { printf(" %.8f", m.wte[i]); }
    printf("\n");

    std::vector<int32_t> tokens = read_bin<int32_t>("models/ref/tokens.bin");
    int T = (int)tokens.size();
    printf("T=%d tokens:", T);
    for (int t = 0; t < T; t++) printf(" %d", tokens[t]);
    printf("\n");

    std::vector<float> x((size_t)T * c.n_embd);
    embed(m, tokens.data(), T, x.data());
    check("embed", x.data(), "models/ref/h00.bin", x.size());

    return 0;
}
