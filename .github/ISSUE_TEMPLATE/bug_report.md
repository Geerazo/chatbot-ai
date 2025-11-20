name: Bug report
description: Reportar un error
labels: [bug]
body:
  - type: textarea
    id: desc
    attributes:
      label: Descripción
      description: ¿Qué ocurre y qué esperabas?
    validations:
      required: true
  - type: textarea
    id: repro
    attributes:
      label: Pasos para reproducir
      placeholder: "1. ...\n2. ..."
  - type: input
    id: version
    attributes:
      label: Versión / entorno
      placeholder: "Python 3.11, Windows 10, commit abc123"
  - type: textarea
    id: logs
    attributes:
      label: Logs (redacta PII)
