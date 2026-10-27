from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


documentos = {
"Politicas de calidad": """

• Estándares Internacionales: La empresa rige su producción bajo la Política de Calidad certificada por IRAM - ISO 9001-2015. 
Esta norma aplica desde el diseño y desarrollo hasta la comercialización de chocolates, pastelería y helados. 
[1] (https://www.rapanui.com.ar/politica).
• Seguridad Alimentaria y TACC: Cuenta con políticas estrictas de segregación en sus plantas 
(como la planta Vereertbrugghen) especializadas en la elaboración de alimentos libres de gluten 
(Sin TACC) para evitar la contaminación cruzada. Sus procesos también siguen lineamientos HACCP e ISO 22000. 
[1] (https://www.rapanui.com.ar/ingredientes-y-productores), [2] (https://www.rapanui.com.ar/politica).
• Abastecimiento Sustentable: Sus políticas de compras exigen que materias primas clave provengan de cooperativas familiares sin agroquímicos (cacao),
huevos de gallinas libres de jaula con certificación Certified Humane, 
y leche de pastoreo libre. 
[1] (https://www.rapanui.com.ar/ingredientes-y-productores)
""",
"Recursos Humanos y Beneficios Laborales":"""

Para sostener una plantilla que supera los 1.600 empleados a nivel global, 
sus políticas de contratación e incentivos en los locales comerciales 
(como los de CABA y Bariloche) estipulan: [1] (https://rapanui.hiringroom.com/jobs), 
[2] (https://www.youtube.com/watch?v=zWNytzaOXJ4&t=761).
• Modalidad de Trabajo: Para puestos operativos en locales (atención, mostrador, heladeros), 
aplican principalmente jornadas part-time con horarios rotativos, orientadas frecuentemente a estudiantes universitarios. 
[1] (https://www.bumeran.com.ar/empleos/dependiente-de-mostrador-rapanui-sucursales-caba-1118411153.html).
• Paquete de Beneficios Corporativos: De acuerdo con sus búsquedas oficiales en portales como Hiring Room y Bumeran, incluyen:
	• Cobertura médica a través de obra social corporativa.
	• 50% de descuento en la red de gimnasios Sport Club.
	• Descuentos exclusivos en productos de la marca.
	• Obsequios corporativos en fechas especiales. [1] (https://www.bumeran.com.ar/empleos/dependiente-de-mostrador-rapanui-sucursales-caba-1118411153.html)
""",
"Comunicación Interna y Cultura Organizacional": """

• Digitalización de Procesos: Debido a la alta dispersión de su personal operativo en fábricas 
y locales que no utilizan computadoras en el día a día, Rapanui adoptó una política de democratización 
digital implementando la plataforma interna Humand. A través de esta aplicación móvil, centralizan la resolución de consultas de RR. HH., 
el envío de recibos de sueldo y las comunicaciones institucionales obligatorias. [1] (https://humand.co/es/caso-de-exito/rapanui-es/).
• Reinversión y Crecimiento: La dirección de la empresa sostiene la política interna 
de reinvertir la totalidad de las utilidades en la propia compañía (en lugar de diversificar en otros mercados financieros) 
para optimizar la eficiencia productiva en el contexto económico argentino. [1] (https://www.youtube.com/watch?v=R4dZSXbcb54)
""",

"Política de Expansión Comercial":
"""
• No Franquiciar: Se mantiene una regla inquebrantable de exclusividad de propiedad. 
Cada punto de venta abierto en Argentina o el exterior es gestionado y financiado de manera directa por la casa central, 
asegurando homogeneidad en la atención, 
la estética de los locales y los precios. [1] (https://www.youtube.com/watch?v=zWNytzaOXJ4&t=761),
[2] (https://www.cronista.com/apertura/inversiones/la-estrategia-de-rapanui-que-va-al-reves-de-los-manuales-y-con-la-que-busca-facturar-600-millones/)

"""
}

def build_vectorstore() -> Chroma:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    persist_dir = base_dir / "vectorstore"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # 1. Si la base ya existe en disco, se reutiliza
    if persist_dir.exists() and any(persist_dir.iterdir()):
        print("Cargando base vectorial persistida desde disco...")
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    # 2. Si no existe, se crean los archivos y se indexa por primera vez
    print("Creando e indexando documentos en disco...")
    data_dir.mkdir(parents=True, exist_ok=True)
    for nombre_archivo, contenido in documentos.items():
        ruta_archivo = data_dir / f"{nombre_archivo}.txt"
        ruta_archivo.write_text(contenido.strip(), encoding="utf-8")
    documentos_crudos = [
        Document(
            page_content=ruta_archivo.read_text(encoding="utf-8"),
            metadata={"source": str(ruta_archivo)},
        )
        for ruta_archivo in data_dir.glob("*.txt")
    ]
    if not documentos_crudos:
        raise RuntimeError(f"No se encontraron documentos para indexar en {data_dir}.")
    split = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=70,
    )
    chunks = split.split_documents(documentos_crudos)
    if not chunks:
        raise RuntimeError("No se generaron fragmentos para crear la base vectorial.")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )