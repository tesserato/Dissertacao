0. resumo

0. abstract

0. listas (figuras, tabelas, siglas)

0. Sumário

1. introdução
	- redes neurais
	- o modelo
    - recorte: simulação de instrumentos **acústicos** (aplicação: bateria)
    - relevância socio-econômica
        - importância social da música
        - vantagens de cultura musical mais acessível
        - benefícios da prática de bateria (psicomotores, stress, etc)
    - áreas de aplicação
        - instrumentos eletronicos (teclados, bateria eletrônica, pianos elétricos)
        - síntese sonora em plataformas como raspberry-pi
        - text to speech mais acessível (acessibilidade)
        - emulação de efeitos de guitarra
        - música eletrônica
2. revisão bibliográfica
    - redes neurais
    - simulação acústica / síntese sonora
    - transformada de Fourier
    - instrumentos musicais eletrônicos
    - acústica aplicada à instrumentos musicais
    - VSTs (instrumentos emulados) -> particularmente o uso de samples x motores de síntese
        - trabalho envolvido
        - tamanho
        - qualidade
        - eficiência computacional
        - verossimilhança (qualidade)
3. metodologia
    - síntese com rede neural pura
        - comparação de arquiteturas(basicamente vanilla x convolucional x recursiva)
            - Com bias: não generaliza bem, mesmo com input = 1
            - Sem bias: mostrar evolução para inputs = 1 , ... , n (erro estgna em algum ponto)  
        - comparação de plataformas(a princípio Python puro x Tensorflow)
    - Pré processamento (via Transformada de Fourier) + (C|R)NN
    - Simulação Física + ajuste via (C|R)NN
    - desenvolvimento do algoritmo de decay (ver modelo)
4. resultados
    - resultado das comparações propostas na metodologia
5. conclusão
    - discussão
    - encaminhamentos futuros
        - desenvolvimento da interface física (ver modelo / pode entrar para o corpo principal se der tempo)

        - aplicação a outros instrumentos

        - MIDI 2.0
---

