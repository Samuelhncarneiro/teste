# app/utils/extraction_monitor.py - SISTEMA DE MONITORAMENTO

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ExtractionStats:
    """Estatísticas de extração para monitoramento"""
    total_pages_processed: int = 0
    successful_pages: int = 0
    failed_pages: int = 0
    total_products_found: int = 0
    products_by_page: List[int] = field(default_factory=list)
    strategy_changes: int = 0
    json_errors: int = 0
    recovery_attempts: int = 0
    recovered_products: int = 0
    expected_products: Set[str] = field(default_factory=set)
    found_products: Set[str] = field(default_factory=set)
    missing_products: Set[str] = field(default_factory=set)
    processing_time: float = 0.0
    pages_with_errors: List[int] = field(default_factory=list)

class ExtractionMonitor:
    """
    Monitor para acompanhar a qualidade da extração e identificar problemas
    """
    
    def __init__(self):
        self.stats = ExtractionStats()
        self.page_details = {}
        
        # Lista de produtos esperados no documento LIUJO
        self.known_liujo_products = {
            "CF5015E0624", "CF5035MS55N", "CF5041MA82O", "CF5042MA82O", 
            "CF5071T4627", "CF5085MS55N", "CF5137D5026", "CF5150MA74Q",
            "CF5180E1030", "CF5203E0624", "CF5206E0624", "CF5215MA75Q",
            "CF5241T3216", "CF5245MS019", "CF5256T073A", "CF5259T054A",
            "CF5271MA96E", "CF5283T4627", "CF5285T4627", "CF5293T4627",
            "CF5302TS869", "CF5305T2589", "CF5317MS019", "CF5345J1857",
            "CF5372T054A", "CF5377T081A", "CF5389MA86Q", "CF5390T2578",
            "CF5394TS74A", "CF5399MA65N", "CF5400MA65N", "CF5412MA820"
        }
        
        self.stats.expected_products = self.known_liujo_products.copy()
    
    def record_page_start(self, page_number: int, strategy: str):
        """Registra o início do processamento de uma página"""
        self.page_details[page_number] = {
            "strategy": strategy,
            "start_time": datetime.now(),
            "products_found": [],
            "errors": [],
            "status": "processing"
        }
    
    def record_page_result(self, page_number: int, result: Dict[str, Any], strategy_used: str):
        """Registra o resultado do processamento de uma página"""
        if page_number not in self.page_details:
            self.page_details[page_number] = {"strategy": strategy_used}
        
        page_detail = self.page_details[page_number]
        page_detail["end_time"] = datetime.now()
        page_detail["final_strategy"] = strategy_used
        
        # Analisar resultado
        if "error" in result:
            # Página com erro
            self.stats.failed_pages += 1
            self.stats.pages_with_errors.append(page_number)
            page_detail["status"] = "failed"
            page_detail["errors"].append(result["error"])
            
            # Identificar tipo de erro
            if "JSON" in result.get("error", ""):
                self.stats.json_errors += 1
                page_detail["error_type"] = "json_parsing"
            
            logger.warning(f"MONITOR: Página {page_number} falhou - {result['error']}")
        else:
            # Página processada com sucesso
            self.stats.successful_pages += 1
            page_detail["status"] = "success"
            
            # Contar produtos encontrados
            products = result.get("products", [])
            products_count = len(products)
            self.stats.products_by_page.append(products_count)
            page_detail["products_count"] = products_count
            
            # Registrar códigos de produtos encontrados
            for product in products:
                material_code = product.get("material_code", "")
                if material_code:
                    page_detail["products_found"].append(material_code)
                    self.stats.found_products.add(material_code)
            
            logger.info(f"MONITOR: Página {page_number} sucesso - {products_count} produtos")
        
        self.stats.total_pages_processed += 1
    
    def record_strategy_change(self, from_strategy: str, to_strategy: str, page_number: int):
        """Registra uma mudança de estratégia"""
        self.stats.strategy_changes += 1
        logger.info(f"MONITOR: Mudança de estratégia na página {page_number}: {from_strategy} → {to_strategy}")
        
        if page_number in self.page_details:
            self.page_details[page_number]["strategy_changed"] = True
            self.page_details[page_number]["original_strategy"] = from_strategy
    
    def record_recovery_attempt(self, page_number: int, recovered_count: int):
        """Registra uma tentativa de recuperação"""
        self.stats.recovery_attempts += 1
        self.stats.recovered_products += recovered_count
        
        if page_number in self.page_details:
            self.page_details[page_number]["recovery_attempted"] = True
            self.page_details[page_number]["recovered_products"] = recovered_count
        
        logger.info(f"MONITOR: Recuperação na página {page_number} - {recovered_count} produtos recuperados")
    
    def finalize_monitoring(self, processing_time: float):
        """Finaliza o monitoramento e calcula estatísticas finais"""
        self.stats.processing_time = processing_time
        self.stats.total_products_found = len(self.stats.found_products)
        self.stats.missing_products = self.stats.expected_products - self.stats.found_products
        
        # Gerar relatório final
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Gera relatório final de monitoramento"""
        logger.info("=" * 60)
        logger.info("RELATÓRIO FINAL DE EXTRAÇÃO")
        logger.info("=" * 60)
        
        # Estatísticas básicas
        logger.info(f"📄 Páginas processadas: {self.stats.total_pages_processed}")
        logger.info(f"✅ Páginas com sucesso: {self.stats.successful_pages}")
        logger.info(f"❌ Páginas com falha: {self.stats.failed_pages}")
        
        if self.stats.total_pages_processed > 0:
            success_rate = (self.stats.successful_pages / self.stats.total_pages_processed) * 100
            logger.info(f"📊 Taxa de sucesso: {success_rate:.1f}%")
        
        # Produtos
        logger.info(f"🛍️ Total de produtos encontrados: {self.stats.total_products_found}")
        logger.info(f"📦 Produtos esperados: {len(self.stats.expected_products)}")
        
        if self.stats.missing_products:
            logger.warning(f"⚠️ Produtos em falta ({len(self.stats.missing_products)}): {sorted(self.stats.missing_products)}")
        
        # Distribuição por página
        if self.stats.products_by_page:
            avg_products = sum(self.stats.products_by_page) / len(self.stats.products_by_page)
            logger.info(f"📈 Média de produtos por página: {avg_products:.1f}")
        
        # Estratégias e recuperação
        if self.stats.strategy_changes > 0:
            logger.info(f"🔄 Mudanças de estratégia: {self.stats.strategy_changes}")
        
        if self.stats.recovery_attempts > 0:
            logger.info(f"🚑 Tentativas de recuperação: {self.stats.recovery_attempts}")
            logger.info(f"💊 Produtos recuperados: {self.stats.recovered_products}")
        
        # Erros
        if self.stats.json_errors > 0:
            logger.warning(f"🔧 Erros de JSON: {self.stats.json_errors}")
        
        if self.stats.pages_with_errors:
            logger.warning(f"📋 Páginas com erro: {self.stats.pages_with_errors}")
        
        # Performance
        logger.info(f"⏱️ Tempo total: {self.stats.processing_time:.2f}s")
        if self.stats.total_products_found > 0:
            products_per_second = self.stats.total_products_found / self.stats.processing_time
            logger.info(f"🚀 Produtos por segundo: {products_per_second:.2f}")
        
        logger.info("=" * 60)
        
        # Alerta para problemas críticos
        self._check_critical_issues()
    
    def _check_critical_issues(self):
        """Verifica e alerta sobre problemas críticos"""
        issues = []
        
        # Taxa de falha muito alta
        if self.stats.total_pages_processed > 0:
            failure_rate = (self.stats.failed_pages / self.stats.total_pages_processed) * 100
            if failure_rate > 30:
                issues.append(f"Taxa de falha alta: {failure_rate:.1f}%")
        
        # Muitos produtos em falta
        missing_rate = len(self.stats.missing_products) / len(self.stats.expected_products) * 100
        if missing_rate > 20:
            issues.append(f"Taxa de produtos perdidos alta: {missing_rate:.1f}%")
        
        # Muitos erros JSON
        if self.stats.json_errors > 2:
            issues.append(f"Muitos erros de JSON: {self.stats.json_errors}")
        
        # Performance muito baixa
        if self.stats.processing_time > 0 and self.stats.total_products_found > 0:
            products_per_second = self.stats.total_products_found / self.stats.processing_time
            if products_per_second < 0.2:
                issues.append(f"Performance baixa: {products_per_second:.2f} produtos/s")
        
        if issues:
            logger.error("🚨 PROBLEMAS CRÍTICOS DETECTADOS:")
            for issue in issues:
                logger.error(f"   - {issue}")
            logger.error("🔧 Recomenda-se revisar configurações e prompts")
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Retorna relatório detalhado para análise"""
        return {
            "stats": {
                "total_pages": self.stats.total_pages_processed,
                "successful_pages": self.stats.successful_pages,
                "failed_pages": self.stats.failed_pages,
                "success_rate": (self.stats.successful_pages / max(1, self.stats.total_pages_processed)) * 100,
                "total_products": self.stats.total_products_found,
                "expected_products": len(self.stats.expected_products),
                "missing_products": len(self.stats.missing_products),
                "recovery_rate": (self.stats.recovered_products / max(1, self.stats.recovery_attempts)) if self.stats.recovery_attempts > 0 else 0,
                "processing_time": self.stats.processing_time,
                "products_per_second": self.stats.total_products_found / max(1, self.stats.processing_time)
            },
            "missing_products": sorted(list(self.stats.missing_products)),
            "found_products": sorted(list(self.stats.found_products)),
            "page_details": self.page_details,
            "strategy_changes": self.stats.strategy_changes,
            "json_errors": self.stats.json_errors,
            "pages_with_errors": self.stats.pages_with_errors
        }
