from psi_lab.cli import main
def test_cli_detecta_picos_y_guarda_figura(tmp_path, capsys):
    salida = tmp_path / "demo.png"
    codigo = main(["-f", "50", "120", "--ruido", "0.1",

    "-o", str(salida)])

    assert codigo == 0
    assert "50.0, 120.0" in capsys.readouterr().out
    assert salida.exists() and salida.stat().st_size > 0